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


from seesaw.dataset import SeesawDataset
import math
import shutil
import torchvision
from transformers import CLIPProcessor, CLIPModel
import roi_extractor
from roi_extractor import AgnosticRoIExtractor
from roi_extractor import to_dataframe
from tqdm import tqdm
#import transforms

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

def preprocess_roi_dataset(
    sds: SeesawDataset,
    output_path,
    clip_model_path = None, 
    cpu=False,
    image_limiter = None, 
    box_limiter = 100,
    padding = 5, 
    start_index = None, 
    end_index = None, 
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

    '''
    vector_path = f"{output_path}/vectors"
    os.makedirs(vector_path)

    model_link = f"{output_path}/model"
    os.symlink(model_path, model_link)

    dataset_link = f"{output_path}/dataset"
    os.symlink(sds.dataset_root, dataset_link)

    real_prefix = f"{os.path.realpath(sds.image_root)}/"
    read_paths = ((real_prefix + sds.paths)).tolist()
    read_paths = [os.path.normpath(p) for p in read_paths]
    meta_dict = dict(zip(read_paths, zip(sds.paths, np.arange(len(sds.paths)))))
    '''

    maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    maskrcnn_model.eval()

    roi_extractor = AgnosticRoIExtractor(maskrcnn_model).to(device)
    roi_extractor.eval()
    roi_extractor.model.rpn.min_size = 10
    roi_extractor.model.rpn.nms_thresh = 0

    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    print("Models defined")
    ims = []
    paths = []
    #excluded = []
    print("Length of Dataset")
    print(len(dataset))
    start = 0
    if start_index != None: 
        start = start_index
    end = len(dataset)
    if end_index != None: 
        end = end_index
    convert_count = 0
    #print(len(dataset))
    with torch.no_grad():
        #for i in tqdm(range(len(dataset))): 
        for i in tqdm(range(start, end)):
            if (i - start) % 2000 == 0: #TURN 87 TO 2000
                if i != start: 
                    print("saving")
                    ans = list(zip(paths, output))
                    #print(output[0]['boxes'])
                    #print(output[0]['new_boxes'])
                    df = to_dataframe(ans)
                    df['dbidx'] = dbidx
                    if clip: 
                        df['clip_feature'] = TensorArray(clip_features)
                    #clip_array = run_clip_on_proposal()
                    #df.assign(clip_feature_vector=TensorArray(clip_array))
                    #print(df.keys())
                    #print(df[['x1', 'y1', 'x2', 'y2', '_x1', '_y1', '_x2', '_y2']])
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
                if data['image'].mode == "L": 
                    print("Converted image: " + str(i))
                    data['image'] = data['image'].convert("RGB")
                    convert_count += 1
                    print(convert_count)
                images = torchvision.transforms.ToTensor()(data['image']).unsqueeze(0).to(device)
                a = roi_extractor(images)[0]
                if a['scores'].shape[0] > box_limiter: 
                    a['boxes'] = torch.split(a['boxes'],box_limiter)[0]
                    a['scores'] = torch.split(a['scores'],box_limiter)[0]
                    a['features'] = torch.split(a['features'].detach(), box_limiter)[0]
                
                #print(data['file_path'])
                #print(a['boxes'])
                clip_array, new_boxes = run_clip_proposal(data['image'], a['boxes'], padding, clip_model, clip_processor, device, i)
                if not isinstance(clip_array, bool): 
                    a['new_boxes'] = torch.tensor(new_boxes).to(device)
                    a['clip_feature_vector'] = clip_array
                    clip_features += clip_array.tolist()
                    output.append(a)
                    dbidx.extend([i]*len(a['scores']))
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



