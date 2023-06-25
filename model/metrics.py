import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from model.utils import read_desc
import numpy as np 
from model.utils import *
import os
import sys
from .evalfunc.bleu.bleu import Bleu
from .evalfunc.rouge.rouge import Rouge
from .evalfunc.cider.cider import Cider
from .evalfunc.meteor.meteor import Meteor



from os.path import join
sys.path.append(join(os.getcwd()))


@torch.no_grad()        
def likelihood(text, tokenizer, model):
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input["labels"] = encoded_input["input_ids"]
    output = model(**encoded_input)
    return output["loss"]


@torch.no_grad()       
def ppl(text, tokenizer, model): 
    """
    
    """

    encodings = tokenizer("\n\n".join(text), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride): 
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]#.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return ppl


def read_gt(dataset, split): 
    gt_fname =f"data/anns/{dataset}_{split}.jsonl"
    gt_data = [ json.loads(line) for line in open(gt_fname).readlines() ]
    ref2gt = {f"{i['image_id']}_REF{i['ref_id']}": [j['sent'] for j in i['sentences']] for i in gt_data}
    return ref2gt 


@torch.no_grad()       
def main(args, pplx=False):

    res_str = ""

    ref2gt = read_gt(args.dataset, args.split)
    ref2cap = read_desc(args.captions_file)
    print(f"LENGTH: GT {len(ref2gt)}; PRED {len(ref2cap)}")
    # scorers expects the prediction as a list of length 1
    ref2cap = {k:[v.lower().replace(".", "")] for k,v in ref2cap.items()}


    # refclef is missing 5 refs 
    if args.dataset == "refclef": 
        ref2gt = {k:v for k,v in ref2gt.items() if k in ref2cap.keys()}
        print(len(ref2gt.keys()), len(ref2cap.keys())) 
        assert(ref2gt.keys() == ref2cap.keys())

    # =================================================
    # Set up scorers
    # =================================================
    print ('setting up scorers...')
    
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Cider(), "CIDEr"),
        (Rouge(), "ROUGE_L"),
    ]
    
    # =================================================
    # Compute scores() in bleu/meteor/cider.py 
    # =================================================
    for scorer, method in scorers:

        score, scores = scorer.compute_score(ref2gt, ref2cap)
        # scs - scores for all refs
        # sc - is the avarage score over all refs in the split
        if type(method) == list:  # BLEU @ X  returns list
            for sc, scs, m in zip(score, scores, method):
                # print ("%s: %0.3f"%(m, sc))
                res_str += "%s: %0.3f, " % (m, sc)
        else:  
            print("%s: %0.3f"%(method, score))
            res_str += "%s: %0.3f, " % (method, score)

    # =================================================
    # Likelihood, Perplexity (PPL) 
    # =================================================     
    print("Computing PPLX...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')   

    ref2cap = read_desc(args.captions_file)
    avg_ppl, avg_likelihood = [],[]
    for tag, text in tqdm(ref2cap.items()):  
        avg_likelihood.append(likelihood(text,tokenizer, model))
        avg_ppl.append(ppl(text, tokenizer, model, device=args.device))

    digits = 3
    avg_ppl = round(np.mean(avg_ppl), digits)
    avg_likelihood = round(np.mean(avg_likelihood), digits)

    print(f"LIKELIHOOD: ", avg_likelihood)
    print(f"PPLX: ", avg_ppl)

    res_str += "%s: %0.3f, " % ("PPL", avg_ppl)
    res_str += "%s: %0.3f, " % ("LIKELIHOOD", avg_likelihood)
    

    print(res_str)
    return res_str
