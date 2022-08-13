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
import tqdm

BEIT_LINK = "/home/gridsan/groups/fastai/omoll/seesaw_root2/models/beit-base-finetuned-ade-640-640"

def process_df(df, multiscale_path, feature_extractor, beit_model): 
    for path in tqdm(df.file_path.unique()): 
        part_df = df[df.file_path == path]
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
            points = set()
            for x in range(int(row['x1']), int(row['x2'])): 
                for y in range(int(row['y1']), int(row['y2'])): 
                    points.add((y, x))
            good = True
            for mask in masks: 
                if points.issubset(mask): 
                    good = False    
                
            if good != True: 
                remove.append(index)
    df = df.drop(remove, axis=0)

    return df

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
    parser.add_argument("--model_path", type=str, required=True, help="path for model")

    args = parser.parse_args()

    import ray
    from seesaw.dataset import SeesawDatasetManager
    from seesaw.indices.multiscale.preprocessor import preprocess_dataset, load_vecs

    ray.init("auto", namespace="seesaw")

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
        print("USING GPU")
    else: 
        device = torch.device("cpu")
        print("Using CPU")
    feature_extractor = BeitFeatureExtractor.from_pretrained(BEIT_LINK)
    beit_model = BeitForSemanticSegmentation.from_pretrained(BEIT_LINK).to(device)

    assert os.path.exists(args.multiscale_path), "multiscale path does not exist"

    parquets = glob.glob(args.multiscale_path + '/vectors.sorted.cached/**.parquet')
    assert len(parquets) != 0, "no parquets detected in vectors.sorted.cached"
    assert not os.path.exists(args.output_path), "output path already exists"

    dirname = os.path.basename(args.output_path)
    dirpath = os.path.dirname(args.output_path)
    output_path = f"{dirpath}/.tmp.{dirname}"
    final_output_path = f"{dirpath}/{dirname}"

    os.makedirs(dirpath, exist_ok=True)

    if os.path.exists(output_path):  # remove old tmpfile
        shutil.rmtree(output_path)

    for parquet in parquets: 
        print("Starting: " + parquet)
        df = pd.read_parquet(parquet)
        new_df = process_df(df, args.multiscale_path, feature_extractor, beit_model)
        new_path = output_path + '/vectors.sorted.cached/' + parquet.split('/')[-1]
        new_df.to_parquet(new_path)

    os.rename(output_path, final_output_path)

    df = load_vecs(args.output_path)
    output_path = resolve_path(args.output_path)
    build_annoy_idx(
        vecs=df["vectors"].to_numpy(),
        output_path=f"{output_path}/vectors.annoy",
        n_trees=100,
    )
