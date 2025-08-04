import numpy as np
import cv2
from PIL import Image
import io

def read_image_as_np(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)
