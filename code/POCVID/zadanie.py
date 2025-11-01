import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.startswith('tm') and filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(img_path)
            images.append(np.array(img))
    return images


images = load_images_from_folder('data')
print(f"Number of images loaded: {len(images)}")
