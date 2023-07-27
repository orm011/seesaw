import einops
import numpy as np
import PIL, PIL.Image
from PIL import Image, ImageDraw, ImageFont
import math

from .multiscale_tools import reconstruct_patch, rearrange_into_tiles, strided_tiling, generate_multiscale_tiling

def create_image_with_text(text):
    # Set image dimensions and background color (white)
    width, height = 224, 224
    background_color = np.random.randint(50,200)

    # Create a NumPy array filled with the background color
    image = np.full((height, width, 3), background_color, dtype=np.uint8)
    image[0,:,:] = 0
    image[:,0,:] = 0
    image[height-1,:,:] = 0
    image[:,width-1,:] = 0

    # Convert the NumPy array to a PIL Image object
    pil_image = Image.fromarray(image)

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Set font properties
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 100)  # Replace "arial.ttf" with the path to your desired font file

    # Get the size of the text to be placed in the image
    (left, top, right, bottom) = draw.textbbox((0,0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    
    # Calculate the position to center the text in the image
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    # Draw the number on the image
    draw.text((x, y), text, font=font, fill=(0, 0, 0))  # Use black color (RGB: 0, 0, 0)
    #draw.rectangle((left, top, right, bottom))
    return pil_image

def make_test_image(b1, b2):
    rb1 = math.ceil(b1)
    rb2 = math.ceil(b2)
 

    test_ims = []
    for i in range(rb1):
        for j in range(rb2):
            test_ims.append(np.array(create_image_with_text(f'{i},{j}')))

    stacked_ims = np.stack(test_ims)
    arr1 = einops.rearrange(stacked_ims, '(rb1 rb2) h w c -> (rb1 h) (rb2 w) c', rb1=rb1)   
    img1 = Image.fromarray(arr1)
    return img1.crop((0, 0, b2*224, b1*224))


def _compare_patches(img1, df1):
    """ check that patch images are equal to cropping the original image using the associated patch metadata """
    for i in range(df1.shape[0]):
        ch0 = PIL.Image.fromarray(df1['tile'].values.to_numpy()[i])
        ch1 = reconstruct_patch(img1, df1.iloc[i])
        assert ch0 == ch1

def test_strided():
    img1 = make_test_image(3,4)
    d1 = strided_tiling(img1, tile_size=224)
    _compare_patches(img1, d1)

def test_full():
    img1 = make_test_image(2,2)
    d1 = generate_multiscale_tiling(img1, factor=.5, tile_size=224)

    assert d1.shape[0] == 4 + 2 + 2 + 1 + 1
    _compare_patches(img1, d1)

def test_num_patches():
    img1 = make_test_image(3,4)
    d1 = rearrange_into_tiles(img1, tile_size=224)
    assert d1['tile'].shape[0] == 12

    img1 = make_test_image(3.3,4.2)
    d1 = rearrange_into_tiles(img1, tile_size=224)
    assert d1['tile'].shape[0] == 12

    img1 = make_test_image(1,1)
    d1 = rearrange_into_tiles(img1, tile_size=224)
    assert d1['tile'].shape[0] == 1