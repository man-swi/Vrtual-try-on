# Data loading and processing
import os
from PIL import Image

def load_image(image_path):
    return Image.open(image_path)

def save_image(image, save_path):
    image.save(save_path)
