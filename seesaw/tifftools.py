import tifffile
import PIL, PIL.Image
import warnings
from PIL import Image
import os
import io
import pandas as pd
import numpy as np


def opentif(fp, path=None) -> PIL.Image.Image:
    import tifffile
    tifim = tifffile.imread(fp)
    t8 = tifim.astype('uint8') 
    if (t8 != tifim).any():
        warnings.warn(f'uint8 conversion does not work losslessly for {path=}')
    asim = Image.fromarray(t8)
    return asim

def batchable(func):
    def wrapper(*args, **kwargs):
        # if the first argument is a list (or more generally, an iterable), 
        # we apply the function to each element
        if isinstance(args[0], pd.DataFrame):
            ans = []
            for tup in args[0].itertuples():
                r = func(tup, *args[1:], **kwargs)
                ans.append(r)
            return pd.DataFrame({'results': np.array(ans)})
        else:
            return func(*args, **kwargs)
    return wrapper
    
def save_image(im, file_path, output_path):
    rel_path = file_path.replace('.tif', '.png')
    full_file_path = f'{output_path}/{rel_path}'
    dir = os.path.dirname(full_file_path)
    os.makedirs(dir, exist_ok=True)
    im.save(full_file_path)

@batchable
def transform_and_save(tup, *, output_folder):
    im = opentif(io.BytesIO(tup.bytes), path=tup.file_path)
    save_image(im, tup.file_path, output_path = output_folder)
    return 1