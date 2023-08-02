import tifffile
import PIL, PIL.Image
import warnings
from PIL import Image
import os
import io
from .util import batchable


def opentif(fp, path=None) -> PIL.Image.Image:
    tifim = tifffile.imread(fp)
    t8 = tifim.astype('uint8') 
    if (t8 != tifim).any():
        warnings.warn(f'uint8 conversion does not work losslessly for {path=}')
    asim = Image.fromarray(t8)
    return asim
    
def save_image(im, file_path, output_path):
    rel_path = file_path.replace('.tif', '.png')
    full_file_path = f'{output_path}/{rel_path}'
    dir = os.path.dirname(full_file_path)
    os.makedirs(dir, exist_ok=True)
    im.save(full_file_path)

@batchable
def transform_and_save(bytes, file_path, *, output_folder):
    im = opentif(io.BytesIO(bytes), path=file_path)
    save_image(im, file_path, output_path = output_folder)
    return 1