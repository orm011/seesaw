from seesaw.definitions import resolve_path
import pandas as pd
from ray.data.extensions import TensorArray
import ray
import io
import torch
import PIL
import os
import PIL.Image
from operator import itemgetter
import numpy as np


from seesaw.dataset import SeesawDatasetManager
import math
import shutil
import torchvision
from transformers import CLIPProcessor, CLIPModel
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation
from tqdm import tqdm
import skimage.measure
#import transforms

BEIT_LINK = "/home/gridsan/groups/fastai/omoll/seesaw_root2/models/beit-base-finetuned-ade-640-640"

def get_beit_boxes(image, feature_extractor, beit_model, device): 
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = beit_model(**inputs)

    logits = torch.nn.functional.interpolate(outputs.logits,
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    # Second, apply argmax on the class dimension
    seg = logits.argmax(dim=1)[0].cpu()
    masks = skimage.measure.label(seg.numpy(), connectivity=1)
    boxes = skimage.measure.regionprops(masks)
    temp_list = []
    for item in boxes: 
        b = item.bbox
        if (abs(b[3] - b[1]) > 10 and abs(b[2] - b[0]) > 10): 
            temp_list.append([b[1], b[0], b[3], b[2]])
    a = [{}]
    n = len(temp_list)
    a[0]['boxes'] = torch.tensor(temp_list).to(device)
    a[0]['num_boxes'] = n
    return a 

def to_dataframe(pairs):

    def to_numpy(d):
        return {k: v.detach().cpu().numpy() for (k, v) in d.items()}

    def box2dict(boxes):
        return {
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
        }

    def paddedBox2Dict(boxes): 
        return {
            "_x1": boxes[:, 0],
            "_y1": boxes[:, 1],
            "_x2": boxes[:, 2],
            "_y2": boxes[:, 3],
        }

    dfs = []
    for (filename, d) in pairs:
        d2 = to_numpy(d)
        rdf = None
        if "new_boxes" in d2.keys(): 
            rdf = pd.DataFrame.from_dict(
                {
                    "filename": filename,
                    **box2dict(d2["boxes"]),
                    **paddedBox2Dict(d2["new_boxes"]),
                }
            )
        else: 
            rdf = pd.DataFrame.from_dict(
                {
                    "filename": filename,
                    **box2dict(d2["boxes"]),
                }
            )
        dfs.append(rdf)

    return pd.concat(dfs, ignore_index=True)

def image_clipper(image, boxes, padding): 
    '''
    This function takes an image, boxes, and the requested padding, and then 
    modifies the boxes to be as square as possible while also having the extra padding
    
    Returns clipped image and new_boxes
    '''
    output = []
    new_boxes = []
    for box in boxes: 
        new_box = box.tolist()
        width = new_box[2] - new_box[0]
        height = new_box[3] - new_box[1]
        diff = abs(height - width)
        hdiff = 0
        vdiff = 0
        if height > width: 
            hdiff = diff + padding
            vdiff = padding
        elif width > height: 
            vdiff = diff + padding
            hdiff = padding
        else: 
            vdiff = padding
            hdiff = padding
        if hdiff != 0: 
            left = max(new_box[0] - round(hdiff/2), 0)
            hdiff -= new_box[0] - left
            right = min(new_box[2] + hdiff, image.size[0])
            hdiff -= right - new_box[2]
            left = max(left - hdiff, 0)
            new_box[0] = left
            new_box[2] = right
        if vdiff != 0: 
            top = max(new_box[1] - round(vdiff/2), 0)
            vdiff -= new_box[1] - top
            bot = min(new_box[3] + vdiff, image.size[1])
            vdiff -= bot - new_box[3]
            top = max(top - vdiff, 0)
            new_box[1] = top
            new_box[3] = bot
        output.append(image.crop(new_box))
        new_boxes.append(new_box)
        
    return output, new_boxes

def run_clip_proposal(image, boxes, padding, clip_model, clip_processor, device, i): 
    '''
    This function takes an image, the boxes, and requested padding and runs the clip embedding on them. 
    Returns the vector of the embedding. 
    '''
    #print("Clip Proposal")
    #print(image)
    images, new_boxes = image_clipper(image, boxes, padding)
    #print(images)
    #print(images)
    #print(images[0])
    #images = torch.tensor(images, dtype=torch.float).to(device)
    try: 
        inputs = clip_processor.feature_extractor(images=images, return_tensors="pt")
    except: 
        print("Missed: " + str(i))
        return False, False
    inputs.to(device)
    vision_outputs = clip_model.vision_model(**inputs)
    image_embeds = vision_outputs[1]
    image_embeds = clip_model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds, new_boxes

def preprocess_beit_dataset(
    sds: SeesawDatasetManager,
    output_path,
    clip_model_path = None, 
    cpu=False,
    image_limiter = None, 
    box_limiter = 100,
    padding = 5, 
):
    if (not cpu) and torch.cuda.is_available(): 
        device = torch.device("cuda")
        print("USING GPU")
    else: 
        device = torch.device("cpu")
        print("Using CPU")
    dataset = sds.get_pytorch_dataset()
    output_path = resolve_path(output_path)
    assert not os.path.exists(output_path), "output path already exists"
    clip = False
    if (clip_model_path): 
        clip = True
        clip_model_path = resolve_path(clip_model_path)
        assert os.path.exists(clip_model_path), "clip model path doesn't exist"

    dirname = os.path.basename(output_path)
    dirpath = os.path.dirname(output_path)
    output_path = f"{dirpath}/.tmp.{dirname}"
    final_output_path = f"{dirpath}/{dirname}"

    os.makedirs(dirpath, exist_ok=True)

    if os.path.exists(output_path):  # remove old tmpfile
        shutil.rmtree(output_path)

    os.makedirs(output_path)

    maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    maskrcnn_model.eval()

    feature_extractor = BeitFeatureExtractor.from_pretrained(BEIT_LINK)
    beit_model = BeitForSemanticSegmentation.from_pretrained(BEIT_LINK).to(device)

    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    print("Models defined")
    ims = []
    paths = []
    #excluded = []
    print("Length of Dataset")
    print(len(dataset))
    start = 40000
    end = len(dataset)
    #print(len(dataset))
    with torch.no_grad():
        #for i in tqdm(range(len(dataset))): 
        for i in tqdm(range(start, end)):
            if i % 2000 == 0: # Keep at 2000
                if i != start: 
                    print("saving")
                    ans = list(zip(paths, output))
                    df = to_dataframe(ans)
                    df['dbidx'] = dbidx
                    if clip: 
                        df['clip_feature'] = TensorArray(clip_features)
                    df.to_parquet(output_path+"/"+str(i)+".parquet")
                clip_features = []
                output = []
                paths = []
                dbidx = []
                
            data = dataset[i]
            if data['image'] is None: 
                print(data)
                #excluded.append(data)

            else: 
                ims.append(data['image'])
                #images = torchvision.transforms.ToTensor()(data['image']).unsqueeze(0).to(device)
                a = get_beit_boxes(data['image'], feature_extractor, beit_model, device)[0]
                num_boxes = a['num_boxes']
                del a['num_boxes']
                if num_boxes > box_limiter: 
                    a['boxes'] = torch.split(a['boxes'],box_limiter)[0]
                    num_boxes = box_limiter
                
                clip_array, new_boxes = run_clip_proposal(data['image'], a['boxes'], padding, clip_model, clip_processor, device, i)
                if not isinstance(clip_array, bool): 
                    a['new_boxes'] = torch.tensor(new_boxes).to(device)
                    a['clip_feature_vector'] = clip_array
                    clip_features += clip_array.tolist()
                    output.append(a)
                    dbidx.extend([i]*num_boxes)
                    paths.append(data['file_path'])
                else: 
                    print("image results were not added")
            
        ans = list(zip(paths, output))
        df = to_dataframe(ans)
        df['dbidx'] = dbidx
        if clip: 
            df['clip_feature'] = TensorArray(clip_features)
        #clip_array = run_clip_on_proposal()
        #df.assign(clip_feature_vector=TensorArray(clip_array))
        df.to_parquet(output_path+"/"+str(i+1)+".parquet")
        #print("EXCLUDED FILES")
        #print(excluded)

        os.rename(output_path, final_output_path)



