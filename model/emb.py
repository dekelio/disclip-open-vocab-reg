import argparse
import clip
import os
import json
from model.utils import Box, build_path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
torch.manual_seed(0)

"""
precompute the clip embedding of box representations.
(1) crop 
(2) blur 
(3) mirror 

Usage: 
    Load with: torch.load("data/embeds/{dataset}_{split}.pt")

"""

def argparser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default="0", help="CUDA device to use.")
    parser.add_argument("--dataset", type=str, default="youreferit")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_dir", type=str, default="data/images/", help="images dir")
    parser.add_argument("--emb_dir", type=str, default="data/embeds")
    args = parser.parse_args()
    
    args.image_dir += args.dataset + "/"
    return args


def adjust_box(old_box, img):
    box = (
    max(old_box.left, 0),
    max(old_box.top, 0),
    min(old_box.right, img.width),
    min(old_box.bottom, img.height)
    )
    return box


def crop(img_pil, box): 
    """
    Syntax: image.crop((left, upper, right, lower))
    |---------------------------|
    |         ^          ^      |
    |       upper        |      |
    |         v          |      |
    |<-left-> |-------|  lower  |
    |         |       |  |      |
    |         |-------|  v      |
    |<---right-------->         |
    |---------------------------|
    """
    return img_pil.crop(box)


def blur(img_pil, box): 
    """ blur the entire image except the box region """
    image = img_pil.copy()
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([box[:2], box[2:]], fill=255)
    blurred = image.filter(ImageFilter.GaussianBlur(100))
    blurred.paste(image, mask=mask)
    return blurred
    

def mirror(img_pil, box): 
    """ mirror padding along the smaller dim. output image is square """
    image = img_pil.copy()
    crop = image.crop(box)
    arr = np.array(crop)
    extra = int(np.abs(arr.shape[0] - arr.shape[1]) / 2)  
    if arr.shape[0] < arr.shape[1]:
        arr = np.pad(arr, ((extra, extra), (0, 0), (0, 0)), mode='reflect')
    else:
        arr = np.pad(arr, ((0, 0), (extra, extra), (0, 0)), mode='reflect')
    return Image.fromarray(arr)


def image_id_to_datum(data):
    id_to_datum = {}
    for datum in data:
        id_to_datum[datum['image_id']] = datum
    return id_to_datum


def crop_with_margin(img_pil, box, ratio=0.2):
    box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    box[0] = box[0] - box[0] * ratio
    box[1] = box[1] - box[1] * ratio
    box[2] = box[2] + box[2] * ratio
    box[3] = box[3] + box[3] * ratio
    box = Box(x=box[0], y=box[1], w=box[2], h=box[3])
    box = adjust_box(box, img_pil)
    return img_pil.copy().crop(box)


def clip_emb(box_representation, model, preprocess): 
    """
    expects pil imahe 
    image_path = build_path(img, args.image_dir)
    img_pil = Image.open(image_path).convert('RGB')
    """
    clip_emb = preprocess(box_representation).unsqueeze(0).to(args.device)
    with torch.no_grad():
        image_embeds = model.encode_image(clip_emb).float()
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds.detach()


def box_emb(image_path, box): 
    img_pil = Image.open(image_path).convert('RGB')
    box_crop = crop(img_pil, box)
    box_blur = blur(img_pil, box)
    box_mirror = mirror(img_pil, box)

    # clip embedding 
    crop_emb = clip_emb(box_crop).detach().cpu()
    blur_emb = clip_emb(box_blur).detach().cpu()
    mirror_emb = clip_emb(box_mirror).detach().cpu()

    return {"crop": crop_emb, "blur":blur_emb, "mirror": mirror_emb}


def get_embeddings(args, save=True): 

    if os.path.exists(args.emb_fname): 
        print(f"Loading precomputed embeddings")       
        embeds = torch.load(args.emb_fname)
        return embeds
    
    print(f"Compute box CLIP embeddings")

    model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
    
    # load boxes from ann file   
    ann_fname = f"data/{args.dataset}_{args.split}.jsonl"

    data = [json.loads(line) for line in open(ann_fname).readlines()]

    image_to_boxes = { datum['image_id']: {'sample': datum["anns"]} for datum in data }

    for img, boxes in image_to_boxes.items():     
        
        image_path = build_path(img, dataset=args.dataset)

        img_pil = Image.open(image_path).convert('RGB')    

        img_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in boxes['sample']]
        
        
        img_crop_emb, img_blur_emb = [],[],[]
        
        for t in img_boxes: 

            box_crop = crop(img_pil, t)
            box_blur = blur(img_pil, t)

            # clip embedding 
            crop_emb = clip_emb(box_crop, model, preprocess).detach().cpu()
            blur_emb = clip_emb(box_blur, model, preprocess).detach().cpu()
            
            img_crop_emb.append(crop_emb)
            img_blur_emb.append(blur_emb)
            
        image_to_boxes[img]['crop'] = img_crop_emb
        image_to_boxes[img]['blur'] = img_blur_emb
        
    # out_fname = f"{args.emb_dir}/{args.dataset}_{args.split}.pt"
    out_fname = args.emb_fname
    torch.save(image_to_boxes, open(out_fname, 'wb+'))
    print(f"Saved {out_fname}")
    return image_to_boxes


if __name__ == "__main__":
    args = argparser()
    torch.set_num_threads(3)
    get_embeddings(args, save=True)

