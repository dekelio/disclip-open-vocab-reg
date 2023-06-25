# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from .utils import Box 
from .utils import read_desc, build_path, set_negset


def write_to_file(text_str, fn):
    f = open(fn, "w")
    f.write(text_str)
    f.close()
    print(f'Saved file to [ {fn} ]')


def adjust_box(box_raw, image):
    box = Box(x=box_raw[0], y=box_raw[1], w=box_raw[2], h=box_raw[3])
    adjusted_box = [
            max(box.left, 0),
            max(box.top, 0),
            min(box.right, image.width),
            min(box.bottom, image.height)]    
    return adjusted_box


def adjust_boxes(image, boxes, target_index):
    
    all_img_boxes = []

    for i in range(len(boxes)):
        box = [
            max(boxes[i].left, 0),
            max(boxes[i].top, 0),
            min(boxes[i].right, image.width),
            min(boxes[i].bottom, image.height)
        ]
        all_img_boxes.append(box)
    
    target_box = all_img_boxes.pop(target_index) 
    distractors = all_img_boxes

    return target_box, distractors


def draw_image_with_bbox(image, pos_box, neg_box, pred_box): 
    draw = ImageDraw.Draw(image)  
    draw.rectangle(pos_box, outline='green', width=4) # #15B01A
    for n in neg_box:
        draw.rectangle(n, outline='red', width=2) 
    
    draw.rectangle(pred_box, outline='blue', width=2) 
    return image


def convert_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def read_gt_sent(dataset, split): 
    gt_fname =f"data/anns/{dataset}_{split}.jsonl"
    gt_data = [json.loads(line) for line in open(gt_fname).readlines()]
    ref2gt = {f"{i['image_id']}_REF{i['ref_id']}": [j['sent'] for j in i['sentences']] for i in gt_data}
    return ref2gt 


def read_gt_cat(dataset, split): 
    gt_fname =f"data/anns/{dataset}_{split}.jsonl"
    gt_data = [json.loads(line) for line in open(gt_fname).readlines()]

    ref2gt = {f"{i['image_id']}_REF{i['ref_id']}": i['category_id'] for i in gt_data}
    
    if "coco" in dataset: 
        coco_cat_labels = coco_obj_categories()
        ref2gt = {k: f"{v} - {coco_cat_labels[v]}" for k,v in ref2gt.items()}
        # obj_cat = f"{cat_id} - {coco_cat_labels[cat_id]}"
    return ref2gt 


def get_ref_info(datum, dataset=""): 
    sent = [ s['sent'] for s in datum['sentences'] ]

    cat_id = datum['category_id'] 
    obj_cat = None
    if "coco" in dataset: 
        coco_cat_labels = coco_obj_categories()
        obj_cat = f"{cat_id} - {coco_cat_labels[cat_id]}"

    return sent, obj_cat


def ref_to_img(data, args, ref2box): 

    image_to_enc = {}

    for imid, ann_to_datum in data.items(): 
        for ann_id, datum in ann_to_datum.items(): 

            rf = datum['ref_id']

            tag = f"{imid}_REF{rf}" 
            if tag not in ref2box: continue
            
            anns = datum['anns']

            # encode image w/ pos neg boxes. 
            image_path = build_path(imid, dataset=args.dataset)
            img_instance = Image.open(image_path).convert('RGB')    
            
            ###
            overlap = False # args.overlap if args.speaker == "magic" else False
            pos_index, neg_index = set_negset(datum, list(ann_to_datum.keys()))
            img_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in anns]
            img_boxes = [ adjust_box(b, img_instance) for b in img_boxes ]
            bbox_pos = img_boxes[pos_index]
            bbox_neg = [ img_boxes[x] for x in neg_index ]
            
            bbox_pred, _ = ref2box[tag]

            if args.listener == "reclip":
                bbox_pred = adjust_box(bbox_pred, img_instance)

            image_with_boxes = draw_image_with_bbox(img_instance, bbox_pos, bbox_neg, bbox_pred)
            image_base64 = convert_image_to_base64(image_with_boxes)
            img_str = '<img src=\'data:image/png;base64,{}\' width=320>'.format(image_base64.decode('utf-8'))
            
            image_to_enc[f"{imid}_REF{rf}"] = img_str

    return image_to_enc # image_to_refs


def coco_obj_categories(): 
    cat_fname = "data/anns/coco-labels-paper.txt"
    cats = [line.replace("\n","") for line in open(cat_fname).readlines()]
    cat = dict(zip(range(1,len(cats)+1), cats))
    return cat


def compose_html(rows, top_text="", nrows=200, grid=0):

    def break_line(grid=False, ngrid=5): 
        if grid: 
            if idx % ngrid == 0 and idx > 0: 
                return "</tr><tr>"
        else: 
            this_img_id = tup[0].split("-")[0].strip()
            same_img = this_img_id == last_img_id
            if (same_img and idx == 1) or not same_img:
                last_img_id = this_img_id
                return "</tr><tr>"
        return ""

    metadata = f"<br>{len(rows)} refs <br><br> <b>Green:</b> target box <br><b>Blue:</b> prediction<br><b>Red:</b> negatives <br>"
    top_text += metadata
    html_code = '<!DOCTYPE html>\n'
    html_code += '<html lang="en"><head><meta charset="utf-8">\n'
    html_code += '<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css">\n'
    html_code +='</head>\n'
    html_code += '<body style="margin:10px;padding:50px">\n'
    html_code += f'<p><br> {top_text} <br></p>\n'
    html_code += '<table border=0 width=95\%>\n'

    html_code += "<tr>"
    last_img_id = 0

    for idx, tup in enumerate(rows):
        
        # limit the number of results printed to file 
        if idx > nrows: 
            print(f"stopped printing to file at {idx}")
            break

        html_code += break_line(grid=True, ngrid=6)
        
        html_code += '<td style="padding:20px">' 
        for i in tup: 
            if type(i) == list:
                html_code += "<b>GT</b>: " + ", ".join(i)
            else:
                html_code += str(i) 
            html_code += "<br>"
        html_code += " </td>\n"

    html_code += "</tr>"    
    html_code += '</table>\n'
    html_code += '</body>\n</html>'
    return html_code

def fix_refs(ref2box, tags): 
    ref_to_tag = {t.split("_REF")[1]: t for t in tags} #t.split("_REF")[0]
    ref2box_new = { ref_to_tag[k]:v for k,v in ref2box.items()}
    return ref2box_new
    

def read_box_prediction(path):
    
    results = [json.loads(line) for line in open(path).readlines()]
    
    is_mcn = "mcn" in path
    
    if is_mcn: 
        ref_to_box = { f"{x['ref_id']}": (x['pred_box'], "") for x in results}
    else: 
        ref_to_box = { f"{x['image_id']}_REF{x['ref_id']}": (x['pred_box'], x['prob']) for x in results}
    return ref_to_box 



def main(data, args, acc):

    # params  = info['params']
    # gt_path =f"data/{args.dataset}_{args.split}.jsonl"
    # gt_data = [json.loads(line) for line in  open(gt_path).readlines()]
    # ref2gt = ref_to_ann(gt_data, args.dataset, args.split, args.image_dir, args.box_fname, args.listener)

    ref2cap = read_desc(args.captions_file)
    ref2sent = read_gt_sent(args.dataset, args.split)
    ref2cat = read_gt_cat(args.dataset, args.split)
    ref2box = read_box_prediction(args.box_fname)

    if args.listener == "mcn": 
        ref2box = fix_refs(ref2box, list(ref2sent.keys()))
    ref2img = ref_to_img(data, args, ref2box)
    tags = [ rf for rf in ref2img.keys() if rf in ref2cap.keys() ]
    # tags = set(ref2img.keys()).intersection(set(ref2cap.keys()))
    
    rows = [ ( str(rf), "<b>Category: </b> " + str(ref2cat[rf]), 
                ref2sent[rf], 
                "<b>Caption: </b> " + ref2cap[rf], 
                "<b>Confidence: </b> " + str(ref2box[rf][1]),
                ref2img[rf]) for rf in tags ]
    
    info = {"speaker":args.speaker, "listener":args.listener, "params": args.captions_file, "accuracy": acc }
    params_txt = "<br>".join([ f"<b>{k}</b>: {v}" for k,v in info.items()])

    html_code = compose_html(rows, top_text=params_txt)
    write_to_file(html_code, args.disp_fname)


