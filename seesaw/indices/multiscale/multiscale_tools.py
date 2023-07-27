import einops
import numpy as np
import PIL, PIL.Image
import math
import pandas as pd


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
    arr = np.array(img1)
    (h, w, _) = arr.shape
    # print(f'{arr.shape=}')
    new_h = (h // tile_size)*tile_size
    new_w = (w // tile_size)*tile_size
    arr = arr[: new_h, :new_w]
    patches = einops.rearrange(arr, '(b1 h) (b2 w) c -> (b1 b2) h w c', h=tile_size, w=tile_size)
    
    ## return metadata
    iis, jjs = np.meshgrid(np.arange(h // tile_size), np.arange(w // tile_size), indexing="ij")
    x1 = jjs.reshape(-1)*tile_size
    y1 = iis.reshape(-1)*tile_size
    x2 = x1 + tile_size
    y2 = y1 + tile_size
    return {'tile':patches, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}


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

    final_res = {}
    for k,_ in all_res[0].items():
        final_res[k] = np.concatenate([r[k] for r in all_res])

    final_res['tile'] = TensorArray(final_res['tile'])
    return pd.DataFrame.from_dict(final_res)

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
    acc = []
    for l in scales:
        df = strided_tiling(l['image'], tile_size=tile_size)
        df['scale_factor'] = l['scale_factor']
        df['zoom_level'] = l['zoom_level']
        df = df.assign(**(df[['x1', 'x2', 'y1', 'y2']]*l['scale_factor']).astype(df['x1'].dtype))
        acc.append(df)
    batch_df = pd.concat(acc, ignore_index=True)
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