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
import roi_extractor
from roi_extractor import AgnosticRoIExtractor
from roi_extractor import to_dataframe
import glob

def image_clipper(image, boxes, padding): 
    '''
    This function takes an image, boxes, and the requested padding, and then 
    modifies the boxes to be as square as possible while also having the extra padding
    
    Returns clipped image and new_boxes
    '''
    output = []
    new_boxes = []
    for box in boxes.numpy(): 
        #print(box.tolist())
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

def run_clip_proposal(image, boxes, padding, clip_model, clip_processor): 
    '''
    This function takes an image, the boxes, and requested padding and runs the clip embedding on them. 
    Returns the vector of the embedding. 
    '''
    images, new_boxes = image_clipper(image, boxes, padding)
    inputs = clip_processor.feature_extractor(images=images, return_tensors="pt")
    vision_outputs = clip_model.vision_model(**inputs)
    image_embeds = vision_outputs[1]
    image_embeds = clip_model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds

def preprocess_roi_dataset(
    sds: SeesawDatasetManager,
    output_path,
    clip_model_path = None, 
    cpu=False,
    image_limiter = None, 
    box_limiter = 100,
    padding = 5, 
):
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

    maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    maskrcnn_model.eval()

    roi_extractor = AgnosticRoIExtractor(maskrcnn_model)
    roi_extractor.eval()

    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

    ims = []
    paths = []
    for i in range(len(dataset)): 
        ims.append(dataset[i]['image'])
        paths.append(dataset[i]['file_path'])
    with torch.no_grad(): 
        images = [torchvision.transforms.ToTensor()(x) for x in ims]
        output = roi_extractor(images)
        
        clip_features = []
        for num, a in enumerate(output): 
            if a['scores'].shape[0] > box_limiter: 
                a['boxes'] = torch.split(a['boxes'],box_limiter)[0]
                a['scores'] = torch.split(a['scores'],box_limiter)[0]
                a['features'] = torch.split(a['features'].detach(), box_limiter)[0]
            if clip: 
                clip_array = run_clip_proposal(ims[num], a['boxes'], padding, clip_model, clip_processor)
                a['clip_feature_vector'] = clip_array
                clip_features += clip_array.tolist()
        ans = list(zip(paths, output))
        df = to_dataframe(ans)
        if clip: 
            df['clip_feature'] = clip_features
        #clip_array = run_clip_on_proposal()
        #df.assign(clip_feature_vector=TensorArray(clip_array))
        df.to_parquet(output_path)


def preprocess_dataset(
    sds: SeesawDatasetManager,
    model_path,
    output_path,
    cpu=False,
    pyramid_factor=0.5,
):
    dataset = sds.get_pytorch_dataset()
    output_path = resolve_path(output_path)
    assert not os.path.exists(output_path), "output path already exists"
    model_path = resolve_path(model_path)
    assert os.path.exists(model_path), "model path doesnt exist"

    dirname = os.path.basename(output_path)
    dirpath = os.path.dirname(output_path)
    output_path = f"{dirpath}/.tmp.{dirname}"
    final_output_path = f"{dirpath}/{dirname}"

    os.makedirs(dirpath, exist_ok=True)

    if os.path.exists(output_path):  # remove old tmpfile
        shutil.rmtree(output_path)

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

    actors = []
    try:
        print("starting actors...")
        ngpus = ray.available_resources().get("GPU", 0)
        ngpus = math.floor(ngpus)

        nactors = ngpus if ngpus > 0 else 1
        actors = [
            ray.remote(Preprocessor)
            .options(num_cpus=5, num_gpus=(1 if ngpus > 0 else 0))
            .remote(jit_path=model_link, output_dir=vector_path, meta_dict=meta_dict)
            for i in range(nactors)
        ]

        rds = ray.data.read_binary_files(
            paths=read_paths, include_paths=True, parallelism=400
        ).split(nactors, locality_hints=actors)

        res_iter = []
        for part_id, (actor, shard) in enumerate(zip(actors, rds)):
            of = actor.extract_meta.remote(shard, pyramid_factor, part_id)
            res_iter.append(of)
        ray.get(res_iter)
        print(f"finished, renaming to {final_output_path}")
        os.rename(output_path, final_output_path)
    finally:
        print("shutting down actors...")
        for a in actors:
            ray.kill(a)


def split_df(df, n_splits):
    lendf = df.shape[0]
    base_lens = [lendf // n_splits] * n_splits
    for i in range(lendf % n_splits):
        base_lens[i] += 1

    assert sum(base_lens) == lendf
    assert len(base_lens) == n_splits

    indices = np.cumsum([0] + base_lens)

    start_index = indices[:-1]
    end_index = indices[1:]
    cutoffs = zip(start_index, end_index)
    splits = []
    for (a, b) in cutoffs:
        splits.append(df.iloc[a:b])

    tot = sum(map(lambda df: df.shape[0], splits))
    assert df.shape[0] == tot
    return splits


def load_vecs(index_path, invalidate=False):
    index_path = resolve_path(index_path)
    ds = SeesawDatasetManager(f"{index_path}/dataset")
    vec_path = f"{index_path}/vectors"
    cached_meta_path = f"{index_path}/vectors.sorted.cached"

    if not os.path.exists(cached_meta_path) or invalidate:
        if os.path.exists(cached_meta_path):
            shutil.rmtree(cached_meta_path)

        tmp_path = f"{index_path}/.tmp.vectors.sorted.cached"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

        idmap = dict(zip(ds.paths, range(len(ds.paths))))

        def assign_ids(df):
            return df.assign(dbidx=df.file_path.map(lambda path: idmap[path]))

        ds = ray.data.read_parquet(
            vec_path,
            columns=["file_path", "zoom_level", "x1", "y1", "x2", "y2", "vectors"],
        )
        df = pd.concat(ray.get(ds.to_pandas_refs()), axis=0, ignore_index=True)
        df = assign_ids(df)
        df = df.sort_values(
            ["dbidx", "zoom_level", "x1", "y1", "x2", "y2"]
        ).reset_index(drop=True)
        df = df.assign(order_col=df.index.values)
        max_zoom_out = df.groupby("dbidx").zoom_level.max().rename("max_zoom_level")
        df = pd.merge(df, max_zoom_out, left_on="dbidx", right_index=True)
        splits = split_df(df, n_splits=32)
        ray.data.from_pandas(splits).write_parquet(tmp_path)
        os.rename(tmp_path, cached_meta_path)
    else:
        print("vecs already exists, reading instead")

    print("reading")
    assert os.path.exists(cached_meta_path)
    ds = ray.data.read_parquet(
        cached_meta_path,
        columns=[
            "dbidx",
            "zoom_level",
            "max_zoom_level",
            "order_col",
            "x1",
            "y1",
            "x2",
            "y2",
            "vectors",
        ],
    )
    df = pd.concat(ray.get(ds.to_pandas_refs()), ignore_index=True)
    assert df.order_col.is_monotonic_increasing, "sanity check"
    return df
    fine_grained_meta = df[["dbidx", "order_col", "zoom_level", "x1", "y1", "x2", "y2"]]
    fine_grained_embedding = df["vectors"].values.to_numpy()
