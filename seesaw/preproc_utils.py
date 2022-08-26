import pandas as pd
from .box_utils import BoundingBoxBatch
import PIL.Image
import io

def decode_image(binary):
    try:
        im = PIL.Image.open(io.BytesIO(binary))
        im.load()
    except:
        im = None    
    return im

def decode_image_batch(input_col, output_col, drop=True):
    ## filters out na images
    def func(df):
        images = df['binary'].map(decode_image)
        if drop:
            df = df.drop(input_col, axis=1)
        df = df.assign(**{output_col:images.values})
        df = df[~images.isna()]
        return df
    
    return func


import ray
def gt_patch_preprocessor(gt_ref):
    def fun(image_df):
        gt_df = ray.get(gt_ref)
        out_batch = pd.merge(image_df, gt_df, left_on='dbidx', right_on='dbidx')
        bbox = BoundingBoxBatch.from_dataframe(out_batch)
        padded_bbox = bbox.pad(padding=20).best_square_box(min_side=160)

        crops = []
        for box,im in zip(padded_bbox.to_xyxy(), out_batch.image.values):
            crop = im.resize((224,244), box=tuple(box), resample=PIL.Image.Resampling.BILINEAR)
            crops.append(crop)

        return out_batch.drop('image', axis=1).assign(crop=crops)
    
    return fun

import torch
from .util import reset_num_cpus
from .models.model import HGFaceWrapper
import torchvision.transforms as T
from ray.data.extensions import TensorArray
import numpy as np
import transformers

class Processor:
    def __init__(self, model_path, input_col, output_col, num_cpus=None):
        import ray

        print(
            f"Init preproc. Avail gpus: {ray.get_gpu_ids()}. cuda avail: {torch.cuda.is_available()}"
        )

        self.device = "cuda:0" if len(ray.get_gpu_ids()) > 0 else "cpu"
        if num_cpus is not None:
            reset_num_cpus(num_cpus)
                
        self.input_col = input_col
        self.output_col = output_col
        self.model = HGFaceWrapper(transformers.CLIPModel.from_pretrained(model_path)).to(self.device)

    def process_dataset(self, ds, part_id):
        dl = ds.window(blocks_per_window=20).map_batches(self.preproc_fun, batch_size=100)
        inference_batches = []
        for batch_df in dl.iter_batches(batch_size=100):
            inf_df = self(batch_df)
            inference_batches.append(inf_df)
        output_df = pd.concat(inference_batches, ignore_index=True)
        
        #self.output_dir = output_dir
        ofile = f"{self.output_dir}/part_{part_id:05d}.parquet"
        output_df.to_parquet(ofile)
        return ofile
    
    def __call__(self, batch_df):
        tensor_xforms=T.Compose([
                lambda im : im.convert('RGB'),
                T.ToTensor(),
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
        ])
        input_ims = batch_df[self.input_col]
        batch = []
        for input_im in input_ims:
            arr = tensor_xforms(input_im)
            batch.append(arr)
        torch_batch = torch.stack(batch).to(self.device)
    
        ### could do some sync preproc here if want guaranteed locality at the cost of synchronous
        with torch.no_grad():
            # vecs = np.ones((batch_df.shape[0], 2))
            vecs = self.model(torch_batch).cpu().numpy().astype('single')

        batch_df = (batch_df.drop(self.input_col, axis=1)
                                .assign(**{self.output_col:TensorArray(vecs)}))
        return batch_df
