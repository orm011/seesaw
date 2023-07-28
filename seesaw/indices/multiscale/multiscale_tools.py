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
    return ims, factors


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

def generate_multiscale_versions(im, factor, min_size):
    """meant to preprocess dict with {path, dbidx,image}"""
    ims, sfs = pyramid(im, factor=factor, abs_min=min_size)
    acc = []
    for zoom_level, (im, sf) in enumerate(zip(ims, sfs), start=1):
        acc.append(
            {
                "image": im,
                "scale_factor": sf,
                "zoom_level": zoom_level,
            }
        )

    return acc

def generate_multiscale_tiling(im, factor, tile_size):
    scales = generate_multiscale_versions(im, factor=factor, min_size=tile_size)

    tiles = []
    acc = []
    max_zoom_level = 1
    for l in scales:
        df = strided_tiling(l['image'], tile_size=tile_size)
        tiles.append(df['tile'].values.to_numpy())
        df['scale_factor'] = l['scale_factor']
        df['scale_factor'] = df['scale_factor'].astype('float32')
        df['zoom_level'] = l['zoom_level']
        max_zoom_level = max(max_zoom_level, l['zoom_level'])
        df = df.assign(scale_factor=df.scale_factor.astype('float32'), zoom_level=df.zoom_level.astype('int16'))
        df = df.assign(**(df[['x1', 'x2', 'y1', 'y2']]*l['scale_factor']).astype('float32'))
        acc.append(df)
    batch_df = pd.concat(acc, ignore_index=True)
    batch_df = batch_df.assign(patch_id=np.arange(batch_df.shape[0], dtype=np.int16), max_zoom_level=max_zoom_level)
    batch_df = batch_df.assign(max_zoom_level=batch_df.max_zoom_level.astype('int16'))

    ## TODO: pick compact types to avoid size blowups
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
def multiscale_preproc_tup(rowtup):
    try:
        image = PIL.Image.open(io.BytesIO(rowtup.bytes))
        tile_df = generate_multiscale_tiling(image, factor=.5, tile_size=224)
    except PIL.UnidentifiedImageError:
        warnings.warn(f'error parsing binary {rowtup.file_path}. Ignoring...')
        tile_df = None
    return tile_df

def multiscale_preproc_batch(batch_df):
    dfs =[]
    for tup in batch_df.itertuples():
        tile_df = multiscale_preproc_tup(tup)
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


def create_multiscale_index(ds, model_path, index_output_path):
    from ray.data import ActorPoolStrategy

    mpath = model_path
    opath = f'{index_output_path}/vectors.sorted.cached'

    ## missing: 
    ### 1. other folder structures: meta json
    rds = ds.as_ray_dataset(parallelism=100)

    (rds.map_batches(multiscale_preproc_batch, batch_format='pandas', batch_size=5)
            .repartition(num_blocks=rds.num_blocks()*10)
            .map_batches(batch_tx, batch_format='pandas', batch_size=200)
            .map_batches(InferenceActor, batch_format='pandas', batch_size=200,
                        compute=ActorPoolStrategy(min_size=1, max_size=2), fn_constructor_kwargs=dict(model_path=mpath), num_gpus=1)
            .repartition(num_blocks=30)
            .write_parquet(opath)
    )