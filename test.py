#!/usr/bin/python

from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
import codecs

# Root directory of the project
ROOT_DIR = os.path.abspath("./")


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import COCO config
# To find local version
sys.path.append(os.path.join(ROOT_DIR, "coco/"))
import coco
# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "h5-models/main.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "public/images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Load a random image from the images folder

results = []

for x in range(1, len(sys.argv)):
    print(sys.argv[x])
    image = skimage.io.imread(os.path.join(
        IMAGE_DIR, sys.argv[x]))
    # Run detection
    result = model.detect([image], verbose=1)
    # Visualize results
    results.append(result[0])


def convert_bitmap_to_array(param):
    b = param.tolist() # nested lists with same data, indices
    print(b)
    return json.dumps(b)

def convert_mask(mask):
    for i in range(0, len(mask)):
        for j in range(0, len(mask[i])):
            indices = [k for k, x in enumerate(mask[i][j]) if x == bool(1)] 
            print(i, j, indices)   

#print(json.dumps({
#    "class_ids": results[0].get("class_ids").tolist(),
#    "scores": convert_bitmap_to_array(results[0].get("scores")),
#    "rois": convert_bitmap_to_array(results[0].get("rois")),
#    "masks":results[0].get("masks").tolist(),
#}))


print(convert_mask(results[0].get("masks").tolist()))