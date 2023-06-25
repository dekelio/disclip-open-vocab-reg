from typing import NamedTuple
import json
from PIL import Image, ImageDraw, ImageFilter
from numpy import argmax
import pandas as pd
import torch
import clip


def detorch(tensor_input):
    return tensor_input.clone().detach().cpu().item()


def flat(l): 
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


class Box(NamedTuple):
    x: int
    y: int
    w: int = 0
    h: int = 0

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center(self):
        return Box(self.x + self.w // 2, self.y + self.h // 2)

    def tolist(self):
        return [self.x , self.y ,self.w, self.h]

    def corners(self):
        yield Box(self.x, self.y)
        yield Box(self.x + self.w, self.y)
        yield Box(self.x + self.w, self.y + self.h)
        yield Box(self.x, self.y + self.h)

    @property
    def area(self):
        return self.w * self.h

    def intersect(self, other: "Box") -> "Box":
        x1 = max(self.x, other.x)
        x2 = max(x1, min(self.x+self.w, other.x+other.w))
        y1 = max(self.y, other.y)
        y2 = max(y1, min(self.y+self.h, other.y+other.h))
        return Box(x=x1, y=y1, w=x2-x1, h=y2-y1)

    def min_bounding(self, other: "Box") -> "Box":
        corners = list(self.corners())
        corners.extend(other.corners())
        min_x = min_y = float("inf")
        max_x = max_y = -float("inf")

        for item in corners:
            min_x = min(min_x, item.x)
            min_y = min(min_y, item.y)
            max_x = max(max_x, item.x)
            max_y = max(max_y, item.y)

        return Box(min_x, min_y, max_x - min_x, max_y - min_y)


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


def normalize(image_embeds):
    return image_embeds / image_embeds.norm(dim=-1, keepdim=True)


def build_path(image_id, dataset="refcoco+", prefix='', file_ending='.jpg'):
    
    image_dir = f"data/images/{dataset}/"
    
    if dataset == "refclef":
        image_id = str(image_id).split("_")[0]
        image_subdir = "".join(["00",image_id.replace(image_id[-3:],"")])[-2:]
        img_path = f"{image_dir}/{image_subdir}/images/{image_id}.jpg"
        return img_path
    
    elif dataset == "refgta":
        return image_dir + str(image_id).replace('final/', '')
        
    elif dataset == "flickr30k": 
        return image_dir + str(image_id) + ".jpg"
    
    img_path = 'COCO_train2014_' 
    padded_ids = str(image_id).rjust(12, '0')
    return image_dir + img_path + prefix + padded_ids + file_ending


def read_crops_caps(fname):
    print(fname)
    imgs_caps = {}
    f = open(fname, "r")
    for line in f.readlines():
        line = line.strip().split(",")
        if line[1] != '':
            if len(line) > 2:
                imgs_caps[line[0]] = ','.join(line[1:]).strip()
            else:
                imgs_caps[line[0]] = line[1].strip()
        else:
            imgs_caps[line[0]] = '<unk>'
    return imgs_caps


def read_desc(fname): 
    df = pd.read_csv(fname,  sep='^([^,]+), ', engine='python', names=["ref", "cap"])
    return dict(zip(df.ref, df.cap)) 


def rearrange_data(data, by='ref_id'): 
    """
    group the data by image_id and ref_id for easy retrival
    """
    image_to_ref = {}
    for datum in data: 
        img_id = datum['image_id']
        if img_id not in image_to_ref.keys(): 
            image_to_ref[img_id] = { datum[by]: datum }
        else: 
            image_to_ref[img_id].update({ datum[by]: datum })
    
    print(f"Organized data: Found {len(data)} refs in {len(image_to_ref)} images")
    
    return image_to_ref


def write_results_to_file(ref2cap, res_fn): 
    f = open(res_fn, "a")
    for rf, cap in ref2cap.items():
        f.write(str(rf) + ", " + str(cap) + "\n")
    f.close()
    print(f"Saved file [ {res_fn} ]")


def read_results(path):

    results = [json.loads(line) for line in open(path).readlines()]

    ref_to_box = {}
    for res in results:
        # TODO
        if res['ref_id'] not in ref_to_box:
            ref_to_box[res['ref_id']] = []
        ref_to_box[res['ref_id']].append(res['pred_box'])    
    return ref_to_box 


def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def compute_acc(box_pred_fname, data, args, thresh=0.5):
    results = read_results(box_pred_fname)
    correct_thresh = 0
    correct_argmax = 0
    img_idx = 0
    length_of_data = 0

    for img_id, ann_to_datum in data.items(): 
        for ann_id, datum in ann_to_datum.items(): 
            
            anns = datum["anns"]
            
            gt_boxes = [[ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]] for ann in anns]
            target_index = next((i for i, item in enumerate(anns) if item["id"] == datum["ann_id"]))
            # target_index = [j for j in range(len(anns)) if anns[j]["id"] == datum[label]][0]
            
            # pred_box = results[img_idx + sent_idx]['pred_box']
            
            rf = datum['ref_id'] if type(list(results.keys())[0]) == int else str(datum['ref_id'])
            if rf not in results: continue

            pred_box = results[rf][0]
            
            # TODO
            if args.listener not in ["mdetr"]:
                pred_box = [pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]]

            gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]

            # iou of each gt box with the pred box
            score = [iou(pred_box, gt_boxes[j]) for j in range(len(gt_boxes))]
            
            if argmax(score) == target_index:
                correct_argmax += 1
            
            # hit if the GT box have iou > 5 with the predicted box. 
            if score[target_index] > thresh:
                correct_thresh += 1
            length_of_data += 1
            
    acc_argmax = (correct_argmax / length_of_data) * 100
    acc_thresh = (correct_thresh / length_of_data) * 100
    return round(acc_argmax,5), round(acc_thresh, 5)


def norm_intersection(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	norm_intersection = interArea / min(boxAArea, boxBArea)
	# return the intersection over union value
	return norm_intersection

    
def read_data(dataset, split): 
    fname = f"data/anns/{dataset}_{split}.jsonl"
    data = [json.loads(line) for line in open(fname).readlines()]
    print(f"Raw Data N = {len(data)} Refs")
    data = rearrange_data(data)
    return data


def set_negset(datum, refs): 
    anns = datum["anns"]
    pos_ann_id = datum["ann_id"]
    neg_ann_id = [ x for x in refs if x != pos_ann_id ]
    pos_index = [i for i in range(len(anns)) if anns[i]["id"] == pos_ann_id][0]
    neg_index = list(range(len(anns)))
    if pos_index in neg_index: neg_index.remove(pos_index)
    return pos_index, neg_index


def clip_emb(object_rep, model, preprocess, device=0): 
    """

    """
    # model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    emb = preprocess(object_rep).unsqueeze(0).to(device).detach()
    with torch.no_grad():
        image_embeds = model.encode_image(emb).float()
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds.detach()


def compute_box_embeds(anns, image_path, device=0, method="crop"): 
    
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    embeds = []
    
    img_pil = Image.open(image_path).convert('RGB')    

    boxes_raw = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in anns]
    boxes = [ adjust_box(b, img_pil) for b in boxes_raw ]

    for x in boxes: 

        if method == "crop": 
            obj_emb = crop(img_pil, x)
            
        if method == "blur": 
            obj_emb = blur(img_pil, x)

        emb = clip_emb(obj_emb, model, preprocess, device=device) 
        embeds.append(emb)

    return embeds


def negset(emb, datum, ann_keys, box_rep="crop-blur", device=0, dataset="refcoco+", precomputed=True): 
    """
    Returns X vectors of dim 512. X - nrefs * n_methods 
    """
    img_id = datum['image_id']

    pos_index, neg_index = set_negset(datum, ann_keys)

    box_embeds = []
    pos_embeds = []
    neg_embeds = []

    # Compute CLIP embedding of all reference boxes (returned in the original order of appearence in the annotations)
    img_path = build_path(img_id, dataset=dataset)

    # convention: rep method are seperated by a hyphen
    for m in box_rep.split("-"): 
        
        if precomputed: 
            box_rep = emb[img_id][m] 
        else:
            box_rep = compute_box_embeds(datum["anns"], img_path, method=m, device=device)
        
        pos = box_rep[pos_index].to(device)
        negs = [ box_rep[x].to(device) for x in neg_index ]
        
        # stack 
        pos_embeds += [pos]
        neg_embeds += negs
        box_embeds = box_embeds + [pos] + negs

    pos_fts = torch.stack(pos_embeds, dim=0).squeeze(1)
    neg_fts = torch.stack(neg_embeds, dim=0).squeeze(1)

    image_embedding = {'pos_fts': pos_fts, 'neg_fts':neg_fts} 
    return image_embedding  # pos_fts, neg_fts 


