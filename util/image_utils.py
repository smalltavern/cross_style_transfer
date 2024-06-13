import pathlib
import numpy as np
from PIL import Image

def load_images(cfg):
    image_style = load_size(cfg.app_image_path)
    image_struct = load_size(cfg.struct_image_path)
    return image_style, image_struct
    
def load_size(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512):
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((size, size)))
    return image