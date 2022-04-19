from transformers import DetrFeatureExtractor, DetrForObjectDetection
from transformers import CLIPProcessor, CLIPModel
import torchvision
import PIL
from PIL import Image, ImageDraw
import requests
import torch
import json
import os
import pandas as pd
import seesaw
from seesaw.imgviz import *
import ray
from seesaw import GlobalDataManager
import importlib

# importlib.reload(seesaw.roi_extractor)
import seesaw.roi_extractor
from seesaw.roi_extractor import AgnosticRoIExtractor
from seesaw.roi_extractor import to_dataframe
import tensorflow as tf
import random
import matplotlib.pyplot as plt


# IMPORT MODELS
maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn_model.eval()

roi_extractor = AgnosticRoIExtractor(maskrcnn_model)
roi_extractor.eval()

clip_model = CLIPModel.from_pretrained(
    "/home/gridsan/groups/fastai/omoll/seesaw_root2/models/clip-vit-base-patch32/"
)
clip_processor = CLIPProcessor.from_pretrained(
    "/home/gridsan/groups/fastai/omoll/seesaw_root2/models/clip-vit-base-patch32/"
)


def run_clip_proposal(image, boxes, padding):
    images, new_boxes = image_clipper(image, boxes, padding)
    inputs = clip_processor.feature_extractor(images=images, return_tensors="pt")
    vision_outputs = clip_model.vision_model(**inputs)
    image_embeds = vision_outputs[1]
    image_embeds = clip_model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds


def image_clipper(image, boxes, padding):
    output = []
    new_boxes = []
    for box in boxes.numpy():
        # print(box.tolist())
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
            left = max(new_box[0] - round(hdiff / 2), 0)
            hdiff -= new_box[0] - left
            right = min(new_box[2] + hdiff, image.size[0])
            hdiff -= right - new_box[2]
            left = max(left - hdiff, 0)
            new_box[0] = left
            new_box[2] = right
        if vdiff != 0:
            top = max(new_box[1] - round(vdiff / 2), 0)
            vdiff -= new_box[1] - top
            bot = min(new_box[3] + vdiff, image.size[1])
            vdiff -= bot - new_box[3]
            top = max(top - vdiff, 0)
            new_box[1] = top
            new_box[3] = bot
        output.append(image.crop(new_box))
        new_boxes.append(new_box)

    return output, new_boxes


def get_proposal_boxes(
    paths, save_path, clip=False, image_limiter=None, box_limiter=100, padding=10
):
    ims = []
    random.shuffle(paths)
    if image_limiter != None:
        ims = [PIL.Image.open(path) for path in paths[: min(image_limiter, len(paths))]]
    else:
        ims = [PIL.Image.open(path) for path in paths]
    with torch.no_grad():
        images = [torchvision.transforms.ToTensor()(x) for x in ims]
        output = roi_extractor(images)

        clip_features = []
        for num, a in enumerate(output):
            if a["scores"].shape[0] > box_limiter:
                a["boxes"] = torch.split(a["boxes"], box_limiter)[0]
                a["scores"] = torch.split(a["scores"], box_limiter)[0]
                a["features"] = torch.split(a["features"].detach(), box_limiter)[0]
            if clip:
                clip_array = run_clip_proposal(ims[num], a["boxes"], padding)
                a["clip_feature_vector"] = clip_array
                clip_features += clip_array.tolist()
        ans = list(zip(paths, output))
        df = to_dataframe(ans)
        if clip:
            df["clip_feature"] = clip_features
        # clip_array = run_clip_on_proposal()
        # df.assign(clip_feature_vector=TensorArray(clip_array))
        df.to_parquet(save_path)
        return df
