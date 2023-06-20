"""
Run this script to prepare the miniImageNet dataset.

This script uses the 100 classes of 600 images each used in the Matching Networks paper. The exact images used are
given in data/mini_imagenet.txt which is downloaded from the link provided in the paper (https://goo.gl/e3orz6).

1. Download files from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view and place in
    data/miniImageNet/images
2. Run the script
"""
import os
import shutil

import numpy as np
from config import DATA_PATH
from few_shot.utils import mkdir, rmdir
from tqdm import tqdm as tqdm

# Clean up folders
rmdir(DATA_PATH + '/miniImageNet/images_background')
rmdir(DATA_PATH + '/miniImageNet/images_evaluation')
mkdir(DATA_PATH + '/miniImageNet/images_background')
mkdir(DATA_PATH + '/miniImageNet/images_evaluation')

# Find class identities
classes = []
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images/'):
    for f in files:
        if f.endswith('.jpg'):
            classes.append(f[:-12])

classes = list(set(classes))
print("classes : ", classes)

# Train/test split
# np.random.seed(0)
# np.random.shuffle(classes)
# background_classes, evaluation_classes = classes[:80], classes[80:]
evaluation_classes = [
    'n01930112', 'n02091831', 'n02108551', 'n02120079', 'n02795169', \
    'n02950826', 'n03075370', 'n04146614', 'n04389033', 'n07613480',\
    'n02089867', 'n02105505', 'n02114548', 'n02174001', 'n02823428', \
    'n02971356', 'n03924679', 'n04296562', 'n04509417', 'n13054560']

background_classes = [c for c in classes if c not in evaluation_classes]

print(len(evaluation_classes), len(background_classes))

# Create class folders
for c in background_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_background/{c}/')

for c in evaluation_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_evaluation/{c}/')

# Move images to correct location
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images'):
    for f in tqdm(files, total=600 * 100):
        if f.endswith('.jpg'):
            class_name = f[:-12]
            image_name = f[-12:]
            # Send to correct folder
            subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
            src = f'{root}/{f}'
            dst = DATA_PATH + f'/miniImageNet/{subset_folder}/{class_name}/{image_name}'
            shutil.move(src, dst)
