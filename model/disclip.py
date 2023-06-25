import torch
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer


class CLIP(nn.Module):
    def __init__(self, method, device, model_name=r"openai/clip-vit-base-patch32"):
        super(CLIP, self).__init__()
        print ('initializing CLIP model...')        
        self.method = method
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def compute_text_representation(self, text_list):
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt",
            max_length=self.tokenizer.max_len_single_sentence + 2, truncation=True)

        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_outputs = self.model.text_model(input_ids=input_ids,attention_mask=attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds

    def compute_image_representation(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        if self.cuda_available:
            pixel_values = pixel_values.cuda(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds

    def compute_image_text_similarity(self, image_embeds, text_list, lam, delta):
        text_list = ["a photo of a " + sen.lower() for sen in text_list]
        text_embeds = self.compute_text_representation(text_list)

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        pos_embeds = image_embeds['pos_fts'].to(self.device)
        neg_embeds = image_embeds['neg_fts'].to(self.device)
        pos_embeds = pos_embeds / pos_embeds.norm(dim=-1, keepdim=True)
        neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
        
        # The number of aug
        nrep = pos_embeds.shape[0]
        logit_scale = self.model.logit_scale.exp()
    
        # pos
        positive_score = torch.matmul(text_embeds, pos_embeds.t()) * logit_scale
        positive_score = 2 * ((1 - delta) * positive_score[:, 0] + delta * positive_score[:, 1])
        
        # negset 
        negatives_score = torch.matmul(text_embeds, neg_embeds.t()) * logit_scale
        idx = int(negatives_score.shape[1] / nrep)
        negatives_score = 2 * ((1 - delta) * negatives_score[:, :idx] + delta * negatives_score[:, idx:])
        negatives_score = torch.mean(negatives_score, dim=1, keepdim=True)

        logits_per_text = lam * positive_score.unsqueeze(1) - (1 - lam) * negatives_score
        probs = logits_per_text.T.softmax(dim=1)
        assert probs.shape[0] == 1 and probs.shape[1] == len(text_list)

        return probs 

