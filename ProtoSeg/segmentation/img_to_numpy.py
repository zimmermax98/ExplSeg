"""
Convert all images to numpy and save (numpy files load much faster than .png images)
"""
import argh
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

from settings import data_path


def convert_images_to_numpy(margin_size: int = 0):
    # for split_key in ['train', 'train_aug', 'val', 'test']:
    for split_key in ['test']:
        img_dir = os.path.join(data_path, f'img_with_margin_{margin_size}/{split_key}')

        if not os.path.exists(img_dir):
            continue

        for filename in tqdm(os.listdir(img_dir), desc=split_key):
            if filename.endswith('.png'):
                img_path = os.path.join(img_dir, filename)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                pix = np.array(img).astype(np.uint8)
                # pix.shape = (height, width, channels)
                np.save(os.path.join(img_dir, '.'.join(filename.split('.')[:-1])), pix)


if __name__ == '__main__':
    argh.dispatch_command(convert_images_to_numpy)
