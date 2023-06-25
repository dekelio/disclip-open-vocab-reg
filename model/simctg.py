import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from .utils import iou
from transformers.generation_logits_process import NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor, \
    NoBadWordsLogitsProcessor, MinLengthLogitsProcessor
from model.disclip import CLIP as DISCLIP

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')

# ========== loss functions ========= #
def compute_valid_token_num(valid_len_list):
    res = 0
    for one_len in valid_len_list:
        res += one_len * (one_len - 1)
    return res

def build_mask_matrix(seqlen, valid_len_list, prefix_len = 0):
    '''
        prefix_len: the length of prefix that we do not want to compute CL loss for.
        (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
            then the loss padding matrix looks like
                 [0., 1., 1., 1.],
                 [1., 0., 1., 1.],
                 [1., 1., 0., 1.],
                 [1., 1., 1., 0.]
        (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
            then the loss padding matrix looks like
                 [0., 1., 1., 0.],
                 [1., 0., 1., 0.],
                 [1., 1., 0., 0.],
                 [0., 0., 0., 0.]
    '''
    res_list = []
    base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
    base_mask = base_mask.type(torch.FloatTensor)
    bsz = len(valid_len_list)
    for i in range(bsz):
        one_base_mask = base_mask.clone()
        one_valid_len = valid_len_list[i]
        one_base_mask[:,one_valid_len:] = 0.
        one_base_mask[one_valid_len:, :] = 0.
        if prefix_len > 0:
            one_base_mask[:prefix_len, :prefix_len] = 0.
        res_list.append(one_base_mask)
    res_mask = torch.stack(res_list, dim = 0)#torch.FloatTensor(res_list)
    
    assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
    return res_mask
        
def contrastive_loss(margin, score_matrix, input_ids, pad_token_id, prefix_len=0):
    '''
       margin: predefined margin to push similarity score away
       score_matrix: bsz x seqlen x seqlen
       input_ids: bsz x seqlen
       pad_token_id: indicating which tokens are padding token
    '''
    bsz, seqlen, _ = score_matrix.size()
    gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1)
    assert gold_score.size() == torch.Size([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix
    assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
    loss_matrix = torch.nn.functional.relu(loss_matrix)

    ### input mask
    input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)
    if loss_matrix.is_cuda:
        input_mask = input_mask.cuda(loss_matrix.get_device())
    input_mask = input_mask.masked_fill(input_ids.eq(pad_token_id), 0.0)

    if loss_matrix.is_cuda:
        input_mask = input_mask.cuda(loss_matrix.get_device())

    valid_len_list = torch.sum(input_mask, dim = -1).tolist()
    loss_mask = build_mask_matrix(seqlen, [int(item) for item in valid_len_list], prefix_len)
    if score_matrix.is_cuda:
        loss_mask = loss_mask.cuda(score_matrix.get_device())
    masked_loss_matrix = loss_matrix * loss_mask

    loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
    assert loss_matrix.size() == input_ids.size()
    loss_matrix = loss_matrix * input_mask
    cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
    return cl_loss

# ========== batch version ========= #
def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            #bsz, num_words, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam // beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def ranking(
    context_hidden, 
    next_hidden, 
    next_top_k_ids, 
    next_top_k_probs, 
    alpha, 
    beta, 
    batch_class_score,
    beam_width):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
        batch_class_score: beam_width x 1
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    scores, _ = torch.max(cosine_matrix, dim = -1)
    next_top_k_probs = next_top_k_probs.view(-1)
    
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores + beta * batch_class_score.view([next_hidden.shape[0]])
    
    scores = torch.stack(torch.split(scores, beam_width))
    selected_idx = scores.max(dim=-1)[1]
    return selected_idx

class SimCTG(nn.Module):
    def __init__(self, **kwargs):
        super(SimCTG, self).__init__()
        # language branch
        from transformers import AutoTokenizer, GPT2LMHeadModel        
        model_name = r'cambridgeltl/magic_mscoco' 
        sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
        eos_token = '<|endoftext|>'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left" # checking

        self.sos_token, self.sos_token_id = self.add_special_token(sos_token)
        print ('sos token is {}, sos token id is {}'.format(self.sos_token, self.sos_token_id))
        self.pad_token, self.pad_token_id = self.add_special_token(pad_token)
        print ('pad token is {}, pad token id is {}'.format(self.pad_token, self.pad_token_id))
        self.eos_token, self.eos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        print ('eos token is {}, eos token id is {}'.format(self.eos_token, self.eos_token_id))
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        print ('Resizing model embedding...')
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        print ('Model embedding resized!')
        self.embed_dim = self.model.config.hidden_size
        self.tokenizer.pad_token = self.tokenizer.eos_token # checking
        self.model.config.pad_token_id = self.model.config.eos_token_id # checking
        
        self.device = kwargs["device"]
        print(f"Running on GPU Device {self.device}")

        # Prevent early completion of caption
        self.sampling_top_k = kwargs["k"]
        self.ending_bonus = kwargs["ending_bonus"]
        # self.use_model = kwargs["use_model"]
        self.gpt_prompt = False  #kwargs["gpt_prompt"]
        end_token = '.'
        self.end_token = self.tokenizer.encode(end_token)[0]
        self.target_seq_length = kwargs["decoding_len"]
        self.prevent_early_finish = MinLengthLogitsProcessor(self.target_seq_length // 2, self.end_token)


    def add_special_token(self, special_token):
        if special_token in self.tokenizer.vocab:
            print (special_token + ' token exists.')
        else:
            print ('Add token to the tokenizer.')
            print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
            self.tokenizer.add_tokens([special_token])
            print ('Vocabulary size after extension is {}'.format(len(self.tokenizer)))
            assert len(self.tokenizer.convert_tokens_to_ids([special_token])) == 1
        special_token_id = self.tokenizer.convert_tokens_to_ids([special_token])[0]
        return special_token, special_token_id

    def initialize_input_ids(self): #, bos_token, img_pil, cats, pos_box, ssa_label):
        
        prompt = self.sos_token + "A photo of a " if self.gpt_prompt else self.sos_token
        # if self.use_model == 'detection':
            # prompt += self.init_sentence_using_object_detection(img_pil, cats, pos_box)
        # if self.use_model == 'classification':
            # prompt += self.init_sentence_using_classifier(img_pil, cats)
        # if self.use_model == 'ssa':
            # prompt += ssa_label
        start_token = self.tokenizer.tokenize(prompt)
        start_token_id = self.tokenizer.convert_tokens_to_ids(start_token)
        input_ids = torch.LongTensor(start_token_id).view(1, -1)
        
        return input_ids.to(self.device)

    @torch.no_grad()
    def get_objects(self, img_pil, cats, score_threshold=0.1): 
        inputs = self.owl_processor(text=cats, images=img_pil, return_tensors="pt").to(self.device)
        outputs = self.owl(**inputs)
        target_sizes = torch.Tensor([img_pil.size[::-1]]).to(self.device)
        results = self.owl_processor.post_process(outputs=outputs, target_sizes=target_sizes)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        boxes, labels = boxes[scores > score_threshold], labels[scores > score_threshold]
        return boxes, labels
    
    def init_sentence_using_object_detection(self, img_pil, cats, pos_box, iou_threshold=0.4):
        boxes, labels = self.get_objects(img_pil, cats)
        if boxes.nelement() == 0:
            return ""
        boxes_ious = [iou(box.tolist(), pos_box) for box in boxes.detach().cpu().numpy()]
        max_iou_idx = np.argmax(boxes_ious)
        max_iou = max(boxes_ious)
        return "" if max_iou < iou_threshold else cats[labels[max_iou_idx].item()]

    def init_sentence_using_classifier(self, img_pil, cats, conf_threshold=0.5):
        img_tensor = pad_to_minmum_size(self.model_size, img_pil).to(self.device).type(torch.cuda.FloatTensor)
        result = self.classifier(img_tensor)
        result = torch.nn.functional.softmax(result, dim=1)
        idx, max_conf = torch.max(result, dim=1)
        return "" if max_conf < conf_threshold else cats[idx.detach().item()]

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels, margin):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = contrastive_loss(margin, cosine_scores, input_ids, self.pad_token_id, prefix_len=0)
        return mle_loss, cl_loss

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum 
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    def parse_sentences(self, text, num_of_sentences_to_keep):
        item_list = text.split('.')
        res_list = item_list[:num_of_sentences_to_keep]
        if len(item_list) > num_of_sentences_to_keep:
            res_text = '.'.join(res_list).strip('.') + '.'
        else:
            res_text = '.'.join(res_list).strip('.').strip()
        return res_text

    def parse_generated_result(self, output, num_of_sentences_to_keep):
        output_text = self.tokenizer.decode(output)
        item_list = output_text.split(self.eos_token)
        full_text = self.eos_token.join(item_list[:2]).strip()
        full_text = self.parse_sentences(full_text, num_of_sentences_to_keep)
        generated_text = item_list[1].strip()
        generated_text = self.parse_sentences(generated_text, num_of_sentences_to_keep)
        return full_text, generated_text

    def parse_output_token_list(self, output):
        output = output.tolist()
        res_list = []
        for token_id in output:
            if token_id == self.sos_token_id:
                continue
            elif token_id == self.eos_token_id:
               break
            else:
                res_list.append(token_id)
        text = self.tokenizer.decode(res_list).strip()
        return ' '.join(text.split()).strip()

    @torch.no_grad()
    def magic_search(self, clip, 
                            beam_width, 
                            alpha, 
                            decoding_len, 
                            beta, 
                            image_embeds,
                            clip_text_max_len, 
                            lam, 
                            delta, 
                            box_representation, 
                            negset=None):
        
        input_ids = self.initialize_input_ids()
        
        prefix_len = input_ids.size()[1]

        past_key_values, last_hidden_states, logits = None, None, None
        generated = [item for item in input_ids.tolist()]
        input_ids_for_class = input_ids.clone()

        # the maximum supported length of generation for SimCTG is 256
        # to support longer generated length, you can re-train the SimCTG model with longer sequences
        if not self.gpt_prompt:
            decoding_len = decoding_len - prefix_len

        for step in range(decoding_len):
            if self.end_token in input_ids: break
            input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class = \
            self.contrastive_decoding(
                self.model, 
                input_ids, 
                prefix_len,
                beam_width, 
                alpha, 
                beta, 
                self.tokenizer,
                image_embeds, 
                clip, 
                clip_text_max_len,
                past_key_values,
                last_hidden_states,
                logits,
                lam,
                delta,
                step,
                box_representation,
                first_step=step==0,
                input_ids_for_class=input_ids_for_class,
                negset=negset
            )
        return self.parse_output_token_list(input_ids_for_class[0])

    
    def contrastive_decoding(self, model, input_ids, prefix_len, beam_width, alpha, beta, 
        simctg_tokenizer, image_embeds, clip, clip_text_max_len, past_key_values, last_hidden_states, 
        logit_for_next_step, lam, delta, i, box_representation, first_step=False, input_ids_for_class=None, negset=None):
        '''
            model: the generation model, e.g., gpt2
            input_ids: 1 x seqlen
        '''

        if first_step:
            output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            past_key_values = output.past_key_values
            last_hidden_states = output.hidden_states[-1]    # [B, S, E]
            logit_for_next_step = output.logits[:, -1, :]    # [B, V]
        
        bsz, seqlen, embed_dim = last_hidden_states.size()
        next_probs = F.softmax(logit_for_next_step, dim = -1)
        _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
        top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)

        # compute the new hidden
        past_key_values = enlarge_past_key_values(past_key_values, beam_width)
        
        output = model(input_ids=top_k_ids.view(-1, 1),
            attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )

        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]
        logits = self.update_special_tokens_logits(input_ids_for_class, i, logits)
        next_hidden = output.hidden_states[-1]
        context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)
        
        # prepare for the classification model
        input_ids_for_class_ = torch.cat([
            input_ids_for_class.unsqueeze(1).expand(-1, beam_width, -1).reshape(bsz*beam_width, seqlen),
            top_k_ids.view(-1, 1)
            ], dim=-1
        )

        batch_text_list = []
        for one_input_id in input_ids_for_class_:
            one_text = simctg_tokenizer.decode(one_input_id[prefix_len:][-clip_text_max_len:]) 
            # we only consider the class score of the generated text continuation
            batch_text_list.append(one_text)
        
        batch_score_crops = clip.compute_image_text_similarity(image_embeds, 
                                                                batch_text_list, 
                                                                lam, 
                                                                delta)

        selected_idx = ranking(context_hidden, 
                                next_hidden, 
                                top_k_ids, 
                                top_k_probs, 
                                alpha, 
                                beta, 
                                batch_score_crops,
                                beam_width)       

        # prepare for the next step
        next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)  
        # print(simctg_tokenizer.decode(next_id[0]))
        
        next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))
        next_hidden = next_hidden[range(bsz), selected_idx, :]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)
        past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
        logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]
        input_ids_for_class = torch.cat([input_ids_for_class, next_id], dim=-1)
        return next_id, past_key_values, last_hidden_states, logits, input_ids_for_class

    
    def update_special_tokens_logits(self, context_tokens, i, logits):
        """
        Source: zerovid/CapGenerator.py
        """
        # Prevent premature ending
        logits = self.prevent_early_finish(context_tokens, logits)

        # Give a tiny constant nudge towards ending, since we tend to end up with too long sentences
        logits[:, self.end_token] += self.ending_bonus

        # Reduce all prob above the ending prob, except those that are still left to consider
        # Like we force prevent short sentences, this promotes ending sentences
        if i >= self.target_seq_length - self.prevent_early_finish.min_length:
            ending_threshold = logits[:, self.end_token].clone()
            # disqualify all logit below the set of active considerations (and reserve a spot for the ending option)
            top_threshold_values, top_threshold_indices = logits.topk(self.sampling_top_k, 1)
            logits[logits < top_threshold_values[:, -1:]] = -np.inf
            logits[:, self.end_token] = ending_threshold
            # Remove the last candidate if ending isn't yet considered
            required_bump = top_threshold_values[:, -1] > ending_threshold
            logits[required_bump, top_threshold_indices[required_bump, -1]] = -np.inf

        # Finally, if somehow a NaN was introduced, set it -inf to remove its impact
        logits[logits != logits] = -np.inf

        return logits