import json
import os
import pandas as pd
from PIL import Image
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path))

from interpreter import *
from executor import *
from methods import *

METHODS_MAP = {
    "baseline": Baseline,
    "random": Random,
    "parse": Parse,
}


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


def read_crops_caps(path):
    imgs_caps = {}
    f = open(path, "r")
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
    """ 
    Expects a file in format 579667_REF198, some text, can be with a comma 
    returns a dict { 45123_REF45: "this is an image of a dog" }
    """
    if not os.path.exists(fname): 
        return {}
    df = pd.read_csv(fname,  sep='^([^,]+), ', engine='python', names=["ref", "cap"])
    return dict(zip(df.ref, df.cap)) 


def read_results(path):
    with open(path) as f:
        lines = f.readlines()       
        results = [json.loads(line) for line in lines]
    ref_to_box = {}
    for res in results:
        if res['ref_id'] not in ref_to_box:
            ref_to_box[res['ref_id']] = []
        ref_to_box[res['ref_id']].append(res['pred_box'])        
    return ref_to_box 


def run(data, captions_file, box_fname, dataset, device_num=0):
    
    print("Starting Listener RECLIP")
    
    method="parse"
    clip_model="ViT-B/32"
    box_representation_method="crop-blur"
    box_method_aggregator="sum"
    blur_std_dev=100
    non_square_size=True
    expand_position_embedding=False

    device = device_num 
    executor = ClipExecutor(clip_model=clip_model, box_representation_method=box_representation_method, method_aggregator=box_method_aggregator, device=device, square_size=non_square_size, expand_position_embedding=expand_position_embedding, blur_std_dev=blur_std_dev, cache_path=None)
    method = METHODS_MAP[method]()

    captions = read_desc(captions_file)
    
    res = []
    ref2box = {}
    
    for (tag, caption) in captions.items():

            img_id, ref_id = tag.split("_REF")
            
            img_path = build_path(img_id, dataset=dataset)
            img = Image.open(img_path).convert('RGB')
            
            anns = data[int(img_id)][int(ref_id)]["anns"]
            gold_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in anns]

            
            for sentence in [caption]:
                boxes = gold_boxes
                env = Environment(img, boxes, executor, False, str(img_id))
                result = method.execute(sentence.lower(), env)
                boxes = env.boxes
                prob = result["probs"][int(result['pred'])]
                dic = {"image_id": img_id,
                        "ref_id": ref_id, 
                        "pred_box": boxes[result["pred"]].tolist(), 
                        "text": result['texts'][0],
                        "prob": str(prob)}
                ref2box[ref_id] = boxes[result["pred"]].tolist()
                res.append(dic)

    f = open(box_fname, "w+")
    for dic in res: 
        json.dump(dic, f)
        f.write("\n")
    f.close()
    print(f"Saved file to [ {box_fname}]")
    
    return ref2box
 
