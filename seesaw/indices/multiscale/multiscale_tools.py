import einops
import numpy as np
import PIL, PIL.Image
import math
import pandas as pd
from ray.data.extensions import TensorArray



def rescale(im, scale, min_size):
    (w, h) = im.size
    target_w = max(math.floor(w * scale), min_size)
    target_h = max(math.floor(h * scale), min_size)
    return im.resize(size=(target_w, target_h), resample=PIL.Image.BILINEAR)

def pyramid(im, factor, abs_min):
    ## example values: factor .71, abs_min 224
    ## if im size is less tha the minimum, expand image to fit minimum
    ## try following: orig size and abs min size give you bounds
    ## returns pyramid from smaller to larger image
    assert factor < 1.0
    factor = 1.0 / factor
    size = min(im.size)
    end_size = abs_min
    start_size = max(size, abs_min)

    start_scale = start_size / size
    end_scale = end_size / size

    ## adjust start scale
    ntimes = math.ceil(math.log(start_scale / end_scale) / math.log(factor))
    start_size = math.ceil(math.exp(ntimes * math.log(factor) + math.log(abs_min)))
    start_scale = start_size / size
    factors = np.geomspace(
        start=start_scale, stop=end_scale, num=ntimes + 1, endpoint=True
    ).tolist()
    ims = []
    for sf in factors:
        imout = rescale(im, scale=sf, min_size=abs_min)
        ims.append(imout)

    assert len(ims) > 0
    assert min(ims[0].size) >= abs_min
    assert min(ims[-1].size) == abs_min
    df = pd.DataFrame({'image':ims, 'scale_factor':factors, 'zoom_level':np.arange(len(ims))})
    return df.sort_values('scale_factor', ascending=True).reset_index(drop=True)

def rearrange_into_tiles(img1, tile_size):
    """
        returns tiles as an array, plus metadata about their origin
    """
    from ray.data.extensions import TensorArray
    arr = np.array(img1)
    (h, w, _) = arr.shape
    # print(f'{arr.shape=}')
    new_h = (h // tile_size)*tile_size
    new_w = (w // tile_size)*tile_size
    arr = arr[: new_h, :new_w]
    patches = einops.rearrange(arr, '(b1 h) (b2 w) c -> (b1 b2) h w c', h=tile_size, w=tile_size)
    
    ## return metadata
    a = np.arange(h//tile_size).astype('int32')
    b = np.arange(w//tile_size).astype('int32')
    ii, jj = np.meshgrid(a,b, indexing='ij')
    ii, jj = ii.reshape(-1), jj.reshape(-1)

    x1 = jj.reshape(-1)*tile_size
    y1 = ii.reshape(-1)*tile_size
    x2 = x1 + tile_size
    y2 = y1 + tile_size
    return pd.DataFrame({'tile':TensorArray(patches), 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2})


def _process_shifted(base_arr, tile_size, shift_y, shift_x):
    s10 = base_arr[shift_y:, shift_x:]
    res = rearrange_into_tiles(s10, tile_size=tile_size)
    res['y1'] = res['y1'] + shift_y
    res['y2'] = res['y2'] + shift_y
    res['x1'] = res['x1'] + shift_x
    res['x2'] = res['x2'] + shift_x
    return res

def strided_tiling(img1, tile_size):
    from ray.data.extensions import TensorArray

    base_arr = np.array(img1)
    stride_size = tile_size //2
    all_res = [] 
    for i in [0,1]:
        for j in [0,1]:
            res = _process_shifted(base_arr, shift_x=stride_size*i, shift_y=stride_size*j, tile_size=tile_size)
            all_res.append(res)

    return pd.concat(all_res, ignore_index=True)

def generate_multiscale_tiling(im, tile_size, factor, min_tile_size):
    pdf = pyramid(im, factor=factor, abs_min=tile_size)

    mask = (224/pdf.scale_factor >= min_tile_size) | (pdf.index == 0) # keep largest at least
    pdf = pdf[mask]
    assert pdf.shape[0] > 0, 'mask eliminated all images. should keep at least one'

    acc = []
    max_zoom_level = pdf.zoom_level.max()
    for tup in pdf.itertuples():
        df = strided_tiling(tup.image, tile_size=tile_size)
        df['scale_factor'] = tup.scale_factor
        df['scale_factor'] = df.scale_factor.astype('float32')
        df['zoom_level'] = tup.zoom_level
        df = df.assign(scale_factor=df.scale_factor.astype('float32'), zoom_level=df.zoom_level.astype('int16'))
        df = df.assign(**(df[['x1', 'x2', 'y1', 'y2']]/tup.scale_factor).astype('float32'))
        acc.append(df)
    batch_df = pd.concat(acc, ignore_index=True)
    batch_df = batch_df.assign(patch_id=np.arange(batch_df.shape[0], dtype=np.int16), max_zoom_level=max_zoom_level)
    batch_df = batch_df.assign(max_zoom_level=batch_df.max_zoom_level.astype('int16'))

    return batch_df

def display_tiles(res):
    from IPython.display import display
    for r in res['tile'].values.to_numpy():
        display(PIL.Image.fromarray(r))

def reconstruct_patch(im1, meta_tup):
    sf = meta_tup.get('scale_factor', 1)
    adjusted_im = rescale(im1, scale=sf, min_size=224)
    ch1 = adjusted_im.crop((math.ceil(meta_tup.x1/sf), math.ceil(meta_tup.y1/sf), math.ceil(meta_tup.x2/sf), math.ceil(meta_tup.y2/sf)))
    return ch1

import warnings
import PIL
from PIL import Image

def opentif(fp, path=None) -> PIL.Image.Image:
    import tifffile
    tifim = tifffile.imread(fp)
    t8 = tifim.astype('uint8') 
    if (t8 != tifim).any():
        warnings.warn(f'uint8 conversion does not work losslessly for {path=}')
    asim = Image.fromarray(t8)
    return asim

import io
def multiscale_preproc_tup(rowtup, min_tile_size):
    try:
        image = PIL.Image.open(io.BytesIO(rowtup.bytes))
        tile_df = generate_multiscale_tiling(image, factor=.5, tile_size=224, min_tile_size=min_tile_size)
    except PIL.UnidentifiedImageError:
        warnings.warn(f'error parsing binary {rowtup.file_path}. Ignoring...')
        tile_df = None
    return tile_df

def multiscale_preproc_batch(batch_df, min_tile_size):
    dfs =[]
    for tup in batch_df.itertuples():
        tile_df = multiscale_preproc_tup(tup, min_tile_size=min_tile_size)
        if tile_df is None:
            continue # empty df messes up types
        
        tile_df = tile_df.assign(**tup._asdict())
        tile_df = tile_df.drop(['bytes', 'Index'], axis=1)
        dfs.append(tile_df)


    ### reorder id columns to show up first.
    cols = ['dbidx', 'file_path', 'patch_id']
    res = pd.concat(dfs, ignore_index=True)

    for i,c in enumerate(cols):
        colval = res[c]
        res = res.drop([c], axis=1)
        res.insert(i, c, colval)
        
    return res


def batch_tx(batch_df):
    import torch
    import torchvision.transforms as T

    if not isinstance(batch_df.tile.values, TensorArray):
        batch_df = batch_df.assign(tile=TensorArray(batch_df.tile.values))

    arr = batch_df.tile.values.to_numpy()
    tmp = einops.rearrange(arr, 'b h w c -> b c h w')
    tmp = torch.from_numpy(tmp).to(torch.float32)
    tmp01 = tmp/255.
    tx = T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)
        )
    normalized = tx(tmp01).to(torch.float32)
    return batch_df.assign(tile=TensorArray(normalized.numpy()))

from seesaw.models.model import ImageEmbedding

class InferenceActor:
    def __init__(self, model_path):
        self.model = ImageEmbedding(device='cuda:0', 
                                    jit_path=model_path, 
                                    add_slide=False)

    def __call__(self, batch_df):
        if not isinstance(batch_df.tile.values, TensorArray):
            batch_df = batch_df.assign(tile=TensorArray(batch_df.tile.values))

        arr = batch_df.tile.values.to_numpy()
        import torch
        tensor = torch.from_numpy(arr).to(self.model.device)
        ans = self.model(preprocessed_image=tensor).to(torch.float32)
        batch_df = batch_df.drop(['tile'], axis=1).assign(vectors=TensorArray(ans.to('cpu').numpy()))
        return batch_df

from seesaw.util import transactional_folder, is_valid_filename
from seesaw.definitions import resolve_path
import json

def run_multiscale_extraction_pipeline(ds, model_path, vector_output_path, min_tile_size):
    from ray.data import ActorPoolStrategy

    rds = ds.as_ray_dataset(parallelism=100)

    (rds.map_batches(multiscale_preproc_batch, batch_format='pandas', batch_size=5, 
                                fn_kwargs=dict(min_tile_size=min_tile_size))
            .repartition(num_blocks=rds.num_blocks()*10)
            .map_batches(batch_tx, batch_format='pandas', batch_size=200)
            .map_batches(InferenceActor, batch_format='pandas', batch_size=200,
                        compute=ActorPoolStrategy(min_size=1, max_size=2), 
                        fn_constructor_kwargs=dict(model_path=model_path), num_gpus=1)
            .repartition(num_blocks=30)
            .write_parquet(vector_output_path)
    )

from seesaw.vector_index import build_annoy_idx

def create_multiscale_index(ds, index_name, model_path, min_tile_size=224, force=False, build_vec_index=False):
    assert is_valid_filename(index_name), index_name

    index_output_path = f'{ds.path}/indices/{index_name}'

    with transactional_folder(index_output_path, force=force) as tmp_output_path:
        model_path = resolve_path(model_path)

        info = {
            "constructor": "seesaw.indices.multiscale.multiscale_index.MultiscaleIndex", 
            "model": model_path, 
            "dataset": resolve_path(ds.path),
        }

        json.dump(info, open(f'{tmp_output_path}/info.json', 'w'), indent=2)

        run_multiscale_extraction_pipeline(ds, model_path=model_path, 
                                       vector_output_path=f'{tmp_output_path}/vectors.sorted.cached',
                                       min_tile_size=min_tile_size
                                       )
        

    # now try loading it
    idx  = ds.load_index(index_name, options=dict(use_vec_index=False))
    if build_vec_index:
         build_annoy_idx(vecs=idx.vectors, output_path=idx.path + '/vectors.annoy', n_trees=10)

    return idx