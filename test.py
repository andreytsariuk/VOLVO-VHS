import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
model_path = os.path.join(ROOT_DIR, "h5-models/tooths.h5")
dataset_path =  os.path.join(ROOT_DIR, "public/images/tooths")
results_dir = os.path.join(ROOT_DIR, "public/images/tooths_result")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# DEFAULT_LOGS_DIR ="./logs"
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"
    DETECTION_MAX_INSTANCES = 10
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # COCO has 80 classes


# IMAGE_MIN_DIM = 1080
# IMAGE_MAX_DIM = 1920




def prepareDatasetAndModel():

    config = CocoConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    # model_path = model.get_imagenet_weights()

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Validation dataset
    #dataset_val = CocoDataset()
    #val_type = "minival"
    #coco = dataset_val.load_coco(dataset_path, val_type, year=2014, return_coco=True, auto_download=False)
    #dataset_val.prepare()
    return  model



#results = getResults(dataset_val, model, coco)
from mrcnn.visualize import *

def display_instances_my(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, pathToSave = None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    print('12')
    # Number of instances
    N = boxes.shape[0]
    my_dpi = 96
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        auto_show = False

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    print('1')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # # Bounding box
        # if not np.any(boxes[i]):
        #     # Skip this instance. Has no bbox. Likely lost in image cropping.
        #     continue
        # y1, x1, y2, x2 = boxes[i]
        # if show_bbox:
        #     p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                         alpha=0.7, linestyle="dashed",
        #                         edgecolor=color, facecolor='none')
        #     ax.add_patch(p)

        # # Label
        # if not captions:
        #     class_id = class_ids[i]
        #     score = scores[i] if scores is not None else None
        #     label = class_names[class_id]
        #     x = random.randint(x1, (x1 + x2) // 2)
        #     caption = "{} {:.3f}".format(label, score) if score else label
        # else:
        #     caption = captions[i]
        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    print('2')
    if (pathToSave is None):
        print('4')
        if auto_show:
            plt.show()
    else:
        print('3')
        plt.savefig(pathToSave, dpi=my_dpi)
        plt.close()
		
import skimage.io



    





def get_size(r):
    TARG_THRESHOLD = 0.9
    classes = r['class_ids']
    scores = r['scores']
    shape = r['masks'].shape
    masks = r['masks'].reshape(shape[2],shape[0],shape[1])
    objectsCount = masks.shape[0]
    
    ma = np.where(masks)
    
    objecstMasks = []
    objectStats = []
    
    toothes = []
    botoms = []
    
    for obId in range(objectsCount):
        objectStats.append(scores[obId] > TARG_THRESHOLD)
        if(classes[obId] == 1):
            toothes.append(obId)
        elif(classes[obId] == 2):
            botoms.append(obId)
    
        curMasksIds = np.where(r['masks'][:, :, obId])
        masksIdsLen = curMasksIds[0].shape[0]
        objectMasks = []
        for i in range(masksIdsLen):
            objectMasks.append([curMasksIds[0][i], curMasksIds[1][i]])
        objecstMasks.append(objectMasks)
    
    # coords = np.nonzero(masks[1])
    # coordsL = list(zip(list(coords[0]), list(coords[1])))
    from sympy import Point, Polygon, Point2D
    from sympy.geometry.util import centroid
    from scipy.spatial import ConvexHull
    
    centroids = []
    toothHights = []
    bottomWidths = []
    
    if(len(toothes) < 2 or len(botoms) == 0):
        print("Bad view point, try again")
    else:
        for id in toothes:
            if(objectStats[id]):
                msk = np.array(objecstMasks[id])
                # maxX = msk[:, 0].max()
                # minX = msk[:, 0].min()
    
                maxY = msk[:, 1].max()
                minY = msk[:, 1].min()
                toothHight = maxY - minY
                toothHights.append(toothHight)
    
                toothMasc = objecstMasks[id]
                import random
    
                toothMasc = random.sample(toothMasc, int(len(toothMasc)/ 15))
                #points = []
                # for pt in toothMasc:
                #     points.append(Point2D(pt[0], pt[1]))
                from sympy.geometry.util import convex_hull
                t = tuple(toothMasc)
               # hull = ConvexHull(t)
                #p = Polygon(*hull.simplices)
                p = Polygon(*t)
                curCentroid = centroid(p)
                centroids.append(curCentroid)
    
        for id in botoms:
            if (objectStats[id]):
    
                toothMasc = objecstMasks[id]
                toothMasc = random.sample(toothMasc, int(len(toothMasc)/ 15))
                t = tuple(toothMasc)
                hull = ConvexHull(t)
                #p = Polygon(*hull.simplices)
                p = Polygon(*t)
                curCentroid = centroid(p)
    
                sumOfDist = 0
                pl = convex_hull(p)
                for simplex in pl.args:
                    curPt = Point2D(simplex)
    
                    sumOfDist += 2 * abs(curPt[1] - curCentroid[1])
                    # size = 2 * sum(abs(p - center) for p in convexhullpoints) / len(convexhullpoints)
    
                bottomWidth = sumOfDist / len(hull.simplices)
                bottomWidths.append(float(bottomWidth))
    
        toothDistances = []
        prevCentr = None
    
        maxId = []
    
        for centr in centroids:
            if(prevCentr is not None):
                dist = abs(centr[0] - prevCentr[0])
                toothDistances.append(float(dist))
    
            prevCentr = centr
    
    
        avgToothDist = np.mean(toothDistances)
        avgToothHigh = np.mean(toothHights)
        meanBottomWidth = np.mean(bottomWidths)
        REAL_DIST_BETWEEN_TOOTH = 72
        mmPerPixel = avgToothDist / REAL_DIST_BETWEEN_TOOTH
        realToothHigth = avgToothHigh / mmPerPixel
        REAL_BOTTOM_H = 10
        mmPerPixel = meanBottomWidth / REAL_BOTTOM_H
        realToothHigth2 = avgToothHigh / mmPerPixel
        #resultH = (realToothHigth + realToothHigth2)
        resultH = realToothHigth
        return resultH





def saveToFile(paths):
    model = prepareDatasetAndModel()

    print('start')
   
    images =[]
    for path in paths:
        images.append(skimage.io.imread(os.path.join(dataset_path, path)))

    res = model.detect(images, verbose=1)
    print('detected')
    for i in range(0,len(res)):
        display_instances_my(images[i], res[i]['rois'], res[i]['masks'], res[i]['class_ids'], ['BG','Tooth','Bottom'], res[i]['scores'], pathToSave = os.path.join(results_dir, paths[i].replace(".jpg",".png")))
    res_1 =  get_size(res[0])
    print(res_1)
    print('saved')





saveToFile([sys.argv[1]])