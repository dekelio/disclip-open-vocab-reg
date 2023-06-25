import argparse
import os


def set_fnames(args): 
    paramstr = paramstring(args)    
    args.captions_file = f"results/{paramstr}.txt"
    args.box_fname = f"results/{args.listener}_{paramstr}.json"
    args.disp_fname = f"results/{args.listener}_{paramstr}.html"
    args.emb_fname = f"data/embeds/{args.dataset}_{args.split}.pt"
    return args


def paramstring(params):
    paramstr = f"{params.speaker}_{params.dataset}_{params.split}_{params.box_representation}_delta_{params.delta}_lam_{params.lam}_k{params.k}_beta_{params.beta}_beam_{params.beam}_negset_{params.negset}_seqlen{params.decoding_len}_end{params.ending_bonus}"
    paramstr = paramstr.replace(".","p")
    return paramstr


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listener",   type=str, 
                                        default="reclip", 
                                        choices="mdetr, reclip",
                                        help="frozen REC model for evaluation")

    parser.add_argument("--speaker", type=str, default="disclip")
    parser.add_argument("--dataset", type=str, default="refcoco+")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use")
    # parser.add_argument("--precompute", action="store_true", help="load precomputed clip-emb of boxes")
    parser.add_argument("--hpt", action="store_true", help="use tuned parames for every dataset")  
    parser.add_argument("--box_representation", type=str, default="crop-blur")
    parser.add_argument("--negset", type=str, default="all", help="negative set. all/hard")

    parser.add_argument("--image_dir", type=str, default=f"data/images/", help="path to images")
    parser.add_argument("--res_dir", type=str, default="") 

    parser.add_argument("--captions_file",  type=str, default="caps.txt") 
    parser.add_argument("--box_fname", type=str, default="box.json")
    parser.add_argument("--emb_fname", type=str, default="data/embeds/")
    
    [args, extras] = parser.parse_known_args()

    # create symlink to the data dirs 
    args.image_dir += args.dataset + "/"
    
    return args, extras




def use_tuned_params(args): 
    """
    Use dataset specific hyper-parameters
    """
    if args.dataset == "refcoco+": 
        args.beta, args.lam = 16.0, 0.6
    if args.dataset == "refcoco": 
        args.beta, args.lam = 20.0, 0.75
    if args.dataset == "refclef": 
        args.beta, args.lam = 20.0, 0.6
    if args.dataset in ["refgta", "flickr30k", "refcocog"]:
        args.beta, args.lam =  32.0, 0.6
    return args 


def make_results_dir(args):

    captions_file_path = f"results/{args.speaker}"
    
    if not os.path.exists(captions_file_path):
        test_text = input("Create dir [ results/ ] ? ")
        ans = int(test_text)
        if ans: 
            os.mkdir(captions_file_path)
        else: 
            print(f"Please create results dir under [ {captions_file_path}  ] ")
            
    box_file_path = f"{os.getcwd()}/results/{args.listener}"

    if not os.path.exists(box_file_path):
        test_text = input(f"Create dir [ results/ {box_file_path} ] ? ")
        ans = int(test_text)
        if ans: 
            os.mkdir(box_file_path)
        else: 
            print(f"Please create results dir under [ {box_file_path}  ] ")
    return captions_file_path, box_file_path
