from model import run as speaker
from model.args import parse_arguments, use_tuned_params, set_fnames
from model.utils import *
from model import metrics
from model.vis import main as disp
import os


def import_listener():
    if args.listener == "reclip": from model.listeners.reclip import main as listener
    elif args.listener == "mdetr": from model.listeners import mdetr as listener
    return listener
        

def set_args(): 

    # read global args
    args, extras = parse_arguments()

    # override and use tuned params if --hpt flag passed 
    if args.hpt: 
        args = use_tuned_params(args)

    # add speaker-specific args 
    args = speaker.get_args(args, extras)
        
    args = set_fnames(args)
    
    return args


if __name__ == "__main__":

    args = set_args()

    data = read_data(args.dataset, args.split)

    # Speaker - Generate REs 
    if not os.path.exists(args.captions_file):
        ref2cap = speaker.run(data, args)

    # Listener - eval w/ frozen REC model
    listener = import_listener()
    if not os.path.exists(args.box_fname):
        ref2box = listener.run(data, args.captions_file, args.box_fname, args.dataset)

    # Compute REC accuracy
    acc_argmax, acc_thresh = compute_acc(args.box_fname, data, args)
    print("ACC: ", "{:10.2f}".format(acc_thresh))

    lang_metrics = metrics.main(args) 

    disp(data, args, acc_thresh)
