import torch
from PIL import Image
import torchvision.transforms as T
import sys
import os
import json


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path))

from ..utils import read_crops_caps, build_path

torch.set_grad_enabled(False)

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def run(data, captions_file, box_fname, dataset, device="cpu"):
    
    print("Starting Listener mDETR")

    # 256 - maximum number of tokens for any given sentence
    model, _ = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
    model = model.to(device)
    model.eval()

    captions = read_crops_caps(captions_file)
    
    res = []
    
    for (tag, caption) in captions.items():
        img_id, ref_id = tag.split("_REF")
        img_path = build_path(img_id, dataset=dataset)
        im = Image.open(img_path).convert('RGB')
        img = transform(im).unsqueeze(0)

        num_of_sentences = 1 

        for idx in range(num_of_sentences):
            
            caption = captions[f"{img_id}_REF{ref_id}"]
            memory_cache = model(img, [caption], encode_and_save=True)
            
            # 'pred_logits', 'pred_boxes', 'proj_queries', 'proj_tokens', 'tokenized'
            outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)
            
            probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1]   
            
            top_box = outputs['pred_boxes'][0, probas.argmax()].unsqueeze(0).to(device)

            bboxes_scaled = rescale_bboxes(top_box, im.size)
            
            top_box_prob = round(float(probas[probas.argmax()]), 2)

            dic = { "image_id": img_id,
                    "ref_id": ref_id, 
                    "pred_box": bboxes_scaled[0].tolist(), 
                    "text": caption,
                    "prob": top_box_prob}

            res.append(dic)
    
    f = open(box_fname, "w+")
    for dic in res: 
        json.dump(dic, f)
        f.write("\n")
    f.close()
    print(f"Saved file to [ {box_fname}]")


