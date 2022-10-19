import argparse
from seesaw.definitions import resolve_path
from seesaw.vector_index import build_annoy_idx
import torch
from torch import nn
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation
import glob
import pandas as pd
import os
import shutil
from PIL import Image
import skimage.measure
from tqdm import tqdm

BEIT_LINK = "/home/gridsan/groups/fastai/omoll/seesaw_root2/models/beit-base-finetuned-ade-640-640"

def process_df(df, multiscale_path, feature_extractor, beit_model): 
    files = df.file_path.unique()
    count = 0
    for path in tqdm(files): 
        temp = 0
        part_df = df[df.file_path == path][['file_path', 'zoom_level', 'x1', 'y1', 'x2', 'y2', 'max_zoom_level']]
        beit_image = Image.open(multiscale_path + '/dataset/images/' + path)
        inputs = feature_extractor(images=beit_image, return_tensors="pt").to(device)
        outputs = beit_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits,
                        size=beit_image.size[::-1], # (height, width)
                        mode='bilinear',
                        align_corners=False)
        seg = logits.argmax(dim=1)[0].cpu() + 1
        masks = skimage.measure.label(seg.numpy(), connectivity=1)
        boxes = skimage.measure.regionprops(masks)
        masks = []
        for box in boxes: 
            l = box.coords.tolist()
            s = set()
            for item in l: 
                s.add(tuple(item))
            masks.append(s)

        remove = []
        for index, row in part_df.iterrows(): 
            good = True
            if row['zoom_level'] != row['max_zoom_level']:
                points = set()
                for x in range(int(row['x1']), int(row['x2'])): 
                    for y in range(int(row['y1']), int(row['y2'])): 
                        points.add((y, x))
                for mask in masks: 
                    if points.issubset(mask): 
                        good = False    
                
            if good != True: 
                count += 1
                temp += 1
                remove.append(index)
        df = df.drop(remove, axis=0)
        print("Dropped {}".format(temp))

    return df, count 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess dataset for use by Seesaw")
    parser.add_argument(
        "--multiscale_path",
        type=str,
        required=True,
        help="where multiscale folder is located",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="where to store the needed output",
    )

    parser.add_argument("--cpu", action="store_true", help="use cpu rather than GPU")
    parser.add_argument("--start", type=int, help="which index to start at")
    parser.add_argument("--end", type=int, help="which index to end at")
    parser.add_argument("--annoy", type=bool, default=False, help="build the annoy")

    args = parser.parse_args()

    import ray
    from seesaw.dataset import SeesawDataset
    from seesaw.indices.multiscale.preprocessor import preprocess_dataset, load_vecs

    #ray.init("auto", namespace="seesaw")

    if args.annoy: 
        df = load_vecs(args.output_path)
        output_path = resolve_path(args.output_path)
        build_annoy_idx(
            vecs=df["vectors"].to_numpy(),
            output_path=f"{output_path}/vectors.annoy",
            n_trees=100,
        )
    else: 
        if torch.cuda.is_available(): 
            device = torch.device("cuda")
            print("USING GPU")
        else: 
            device = torch.device("cpu")
            print("Using CPU")
        feature_extractor = BeitFeatureExtractor.from_pretrained(BEIT_LINK)
        beit_model = BeitForSemanticSegmentation.from_pretrained(BEIT_LINK).to(device)

        assert os.path.exists(args.multiscale_path), "multiscale path does not exist"

        parquets = sorted(glob.glob(args.multiscale_path + '/vectors.sorted.cached/**.parquet'))
        assert len(parquets) != 0, "no parquets detected in vectors.sorted.cached"
        os.makedirs(args.output_path, exist_ok=True)

        output_path = args.output_path

        start = 0
        if args.start != None: 
            start = args.start 
        end = len(parquets)
        if args.end != None: 
            end = args.end 
        print("Number of Parquets: {}".format(len(parquets)))
        print("Started at: {}, Ending at: {}".format(start, end))
        total = 0
        for parquet in parquets[start:end]: 
            print("Starting: " + parquet)
            df = pd.read_parquet(parquet)
            new_df, temp = process_df(df, args.multiscale_path, feature_extractor, beit_model)
            total += temp
            new_path = output_path + 'vectors.sorted.cached/' + parquet.split('/')[-1]
            os.makedirs(output_path + 'vectors.sorted.cached/', exist_ok=True)
            new_df.to_parquet(new_path)
            print("Saved " + new_path)
        print("Total boxes removed: {}".format(total))


    
