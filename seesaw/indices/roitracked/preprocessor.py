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
from roi_track_extractor import AgnosticRoIExtractor
from roi_track_extractor import to_dataframe
from tqdm import tqdm

from deepsort import Detection, NearestNeighborDistanceMetric, Tracker
#import transforms

# Definition of the parameters
max_cosine_distance = 0.7
max_euclidean_distance = 0.7
nn_budget = None

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

def run_clip_proposal(image, boxes, padding, clip_model, clip_processor, device): 
    '''
    This function takes an image, the boxes, and requested padding and runs the clip embedding on them. 
    Returns the vector of the embedding. 
    '''
    images, new_boxes = image_clipper(image, boxes, padding)
    inputs = clip_processor.feature_extractor(images=images, return_tensors="pt")
    inputs.to(device)
    vision_outputs = clip_model.vision_model(**inputs)
    image_embeds = vision_outputs[1]
    image_embeds = clip_model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds, new_boxes

def preprocess_roi_dataset(
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
    start = 0
    end = len(dataset)
    last_track_id = None
    tracker = None
    with torch.no_grad():
        #for i in tqdm(range(len(dataset))): 
        for i in tqdm(range(start, end)):
            if i % 2000 == 0: 
                if i != start: 
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
                paths.append(data['file_path'])
                images = torchvision.transforms.ToTensor()(data['image']).unsqueeze(0).to(device)
                a = roi_extractor(images)[0]
                if a['scores'].shape[0] > box_limiter: 
                    a['boxes'] = torch.split(a['boxes'],box_limiter)[0]
                    a['scores'] = torch.split(a['scores'],box_limiter)[0]
                    a['features'] = torch.split(a['features'].detach(), box_limiter)[0]
                dbidx.extend([i]*len(a['scores']))
                if clip: 
                    clip_array, new_boxes = run_clip_proposal(data['image'], a['boxes'], padding, clip_model, clip_processor, device)
                    a['new_boxes'] = torch.tensor(new_boxes).to(device)
                    a['clip_feature_vector'] = clip_array
                    
                    clip_features += clip_array.tolist()
                track_id = data['file_path'].split('/')[0]
                if track_id != last_track_id: 
                    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
                    tracker = Tracker(metric)
                    last_track_id = track_id
                detection_list = []
                for j in range(a['scores'].shape[0]): 
                    var = a['boxes'][j]
                    box = [var[0].item(), var[1].item(), abs(var[2] - var[0]).item(), abs(var[3] - var[1]).item()]
                    det = Detection(box, a['scores'][j], 'seesaw', a['clip_feature_vector'][j])
                    detection_list.append(det)
                matches = object_tracking(detection_list, tracker)
                a['track_id'] = torch.tensor(matches)
                output.append(a)
            
        ans = list(zip(paths, output))
        df = to_dataframe(ans)
        df['dbidx'] = dbidx
        if clip: 
            df['clip_feature'] = TensorArray(clip_features)
        df.to_parquet(output_path+"/"+str(i+1)+".parquet")

        os.rename(output_path, final_output_path)


# Function for object tracking on video
def object_tracking(detection_list, tracker):
    
    
    # Pass detections to the deepsort object and obtain the track information.
    tracker.predict()
    matches = tracker.update(detection_list)

    return matches



