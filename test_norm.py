import numpy as np
import os

p = os.path.expanduser("~/Clara/public_dataset_npy")
for root, _, files in os.walk(p):
    masks = [f for f in files if f.endswith('_mask.npy')]
    imgs = [f for f in files if f.endswith('_img.npy')]
    if masks and imgs:
        mask = np.load(os.path.join(root, masks[0]))
        img = np.load(os.path.join(root, imgs[0]))
        print(f"Mask unique values: {np.unique(mask)}")
        print(f"Image min: {img.min()}, max: {img.max()}")
        break
