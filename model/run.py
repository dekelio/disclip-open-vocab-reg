import argparse
from .emb import get_embeddings
from .simctg import SimCTG  
from .disclip import CLIP as DISCLIP 
from .utils import *

"""

"""

def get_args(global_args, global_extras):
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, default=0.1, help="degeneration penalty")
    parser.add_argument("--beta", type=float, default=4.0, help="trade-off parameter vision and language")
    parser.add_argument("--delta", type=float, default=0.5, help="trade-off parameter crop and blur. 0=crop")
    parser.add_argument("--lam", type=float, default=0.75, help="trade-off parameter pos box and negs boxes.")
    
    parser.add_argument("--decoding_len", type=int, default=16, help="Length of sentence to generate by LM")
    parser.add_argument("--clip_max_len", type=int, default=60, help="Max Length of sentence to generate by CLIP")

    parser.add_argument("--k", type=int, default=45, help="number LM candidates")
    parser.add_argument("--beam", type=int, default=1)
    
    parser.add_argument("--ending_bonus", type=float, default=0.0, help='How much to help the sentence to end')
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help='How much much to deter deter repeats')

    parser.add_argument("--detection", default=False, action='store_true')
    parser.add_argument("--overlap", default=False, action='store_true', help="remove negs with high overlap")
    
    params = parser.parse_args(global_extras, namespace=global_args)
    
    return params 


def get_dis_cap(data, args):
    
    print(f"Intializing speaker")
    text_generator = SimCTG(**vars(args))
    text_generator.to(args.device)
    text_generator.eval()

    clip = DISCLIP(args.box_representation, args.device)
    clip.to(args.device)
    clip.eval()
    
    # pre-compute clip embeddings to eccelerate inference
    embeds = get_embeddings(args)

    ref2cap = {}
 
    for img_id, ann_to_datum in data.items(): 
        for _, datum in ann_to_datum.items(): 
            
            img_id = datum['image_id']
            ref_id = datum['ref_id']
            tag = f"{img_id}_REF{ref_id}"

            # pos_fts, neg_fts 
            image_embedding = negset(embeds, datum, list(ann_to_datum.keys()),  
                                    box_rep=args.box_representation, 
                                    device=args.device, 
                                    dataset=args.dataset)    

            cap = text_generator.magic_search(clip, args.k,
                                    args.alpha,
                                    args.decoding_len,
                                    args.beta,
                                    image_embedding,
                                    args.clip_max_len,
                                    args.lam,
                                    args.delta,
                                    args.box_representation, 
                                    negset=args.negset)                                                    
            
            cap = cap.replace("A photo of a ", "")
            ref2cap[tag] = cap
            print(tag, cap)
    return ref2cap


def run(data, args): 
    print(f"Start REG")
    ref2cap = get_dis_cap(data, args)
    write_results_to_file(ref2cap, args.captions_file)
    return ref2cap
