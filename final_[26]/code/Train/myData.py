"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import yaml
import json
import datetime
import random
import numpy as np
import skimage.draw
import cv2
import imgaug.augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################
class TargetsConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "targets"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    #     IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    RPN_NMS_THRESHOLD = 0.9
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + joint

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.995

    BACKBONE = "resnet50"
    DETECTION_NMS_THRESHOLD = 0.4
    RPN_ANCHOR_SCALES = (128, 256, 512)

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        "mrcnn_mask_edge_loss": 1.
    }
    EDGE_LOSS_FILTERS = ["sobel-x", "sobel-y"]  #
    EDGE_LOSS_WEIGHT_FACTOR = 0.5
    EDGE_LOSS_WEIGHT_ENTROPY = True


############################################################
#  Dataset
############################################################
class TargetsDataset(utils.Dataset):
    def __init__(self, gray=False, train_split=None, val_dataset=False, read_depth=False, input_RGBD=True):
        super().__init__(class_map=None)
        self.gray = gray
        self.train_split = train_split
        self.val_dataset = val_dataset
        self.read_depth = read_depth
        self.input_RGBD = input_RGBD

    def load_targets(self, dataset_dir, xmin, xmax, ymin, ymax):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 2 classes to add.
        self.add_class("targets", 1, "joint")
        print('dataset_dir:', dataset_dir)
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        annotations = sorted(annotations, key=lambda x: int(x['filename'].split('_')[0]))

        random.seed(0)
        random.shuffle(annotations)
        self.annotations = annotations

        # Add images
        # split from train or validation dataset
        if self.train_split:
            to_train = int(len(self.annotations) * self.train_split)
            # return dataset for validation ?            
            the_annotations = annotations[to_train:] if self.val_dataset else annotations[:to_train]
        else:
            the_annotations = annotations

        for a in the_annotations:
            # Get the x, y coordinaets of points of the polygons that make up the outline of each object instance.
            # These are stores in the shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x (dict)and 2.x (list).
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                cls = [r['region_attributes']['type'] for r in a['regions'].values()]

            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                cls = [r['region_attributes']['type'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read the image.
            # This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            print('val data', image_path) if self.val_dataset else print('train data', image_path)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            if xmin is None: xmin = 0
            if xmax is None: xmax = width
            if ymin is None: ymin = 0
            if ymax is None: ymax = height

            self.add_image(
                "targets",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                cls=cls
            )

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        if self.input_RGBD:
            img_path = self.image_info[image_id]['path']  # RGB image path
            depth_path = img_path[:-7] + 'depth.png'
            # read RGB image
            rgb_image = skimage.io.imread(img_path)[self.image_info[image_id]['ymin']:self.image_info[image_id]['ymax'],
                        self.image_info[image_id]['xmin']:self.image_info[image_id]['xmax']]
            if rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[..., :3]

            # read depth image and inpaint
            depth_image = skimage.io.imread(depth_path)[
                          self.image_info[image_id]['ymin']:self.image_info[image_id]['ymax'],
                          self.image_info[image_id]['xmin']:self.image_info[image_id]['xmax']]
            depth_image = depth_image[..., None].repeat(repeats=3, axis=2)
            # inpainting
            depth_img_base = (cv2.copyMakeBorder(depth_image, 1, 1, 1, 1, cv2.BORDER_DEFAULT))[..., :1]

            mask = (depth_img_base == 0).astype(np.uint8)
            scale = np.abs(depth_img_base).max()
            depth_img_base = depth_img_base.astype(np.float32) / scale  # Has to be float32, 64 not supported.
            depth_img_base = cv2.inpaint(np.float32(depth_img_base), mask, 1, cv2.INPAINT_NS)
            # Back to original size and value range.
            depth_img_base = depth_img_base[1:-1, 1:-1]
            depth_image = depth_img_base  # * scale

            # concate RGBD
            depth_image = np.expand_dims(depth_image, axis=2)
            image = np.concatenate((rgb_image, depth_image), axis=2)

        else:
            if self.read_depth:
                # because of the data collection rule: time_RGB.png for rgb image, time_Depth.png for depth image.
                img_path = self.image_info[image_id]['path'][:-7] + 'Depth.png'
            else:
                img_path = self.image_info[image_id]['path']

            image = skimage.io.imread(img_path)
            image = image[self.image_info[image_id]['ymin']:self.image_info[image_id]['ymax'],
                    self.image_info[image_id]['xmin']:self.image_info[image_id]['xmax']]

            if self.read_depth:
                # img = self.depth_inpaint(image)
                depth_img_base = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
                mask = (depth_img_base == 0).astype(np.uint8)
                scale = np.abs(depth_img_base).max()
                depth_img_base = depth_img_base.astype(np.float32) / scale  # Has to be float32, 64 not supported.
                depth_img_base = cv2.inpaint(np.float32(depth_img_base), mask, 1, cv2.INPAINT_NS)
                # Back to original size and value range.
                depth_img_base = depth_img_base[1:-1, 1:-1]
                image = depth_img_base  # * scale

            if self.gray:  # or self.read_depth:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                image = np.stack((image,) * 3, axis=-1)
                print('gray size', image.shape)

            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
        return image

    def load_mask(self, image_id, occluded=True):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        if image_info["source"] != "targets":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        # Since we have one class ID only, we return an array of 1s
        # print(self.annotations[0])#["region_attributes"])
        # print(image_info["class"])
        classes = list(map(int, image_info["cls"]))

        masks = mask.astype(np.bool)[info['ymin']:info['ymax'], info['xmin']:info['xmax']]
        class_ids = np.array(classes, dtype=np.int32)  # np.ones([mask.shape[-1]], dtype=np.int32)

        if occluded:
            masks = self.get_occluded_mask(masks)

        return masks, class_ids

    def get_occluded_mask(self, mask):
        """
        Generate occluded instance masks for an image according to the label sequence.
        Returns: occluded_mask: A bool array of shape [height, width, instance count] with one mask per instance.
        """

        occluded_mask = np.zeros(mask.shape, dtype=np.uint8)
        for i in range(mask.shape[2]):
            m0 = mask[:, :, i]
            m00 = m0.copy()

            for j in range(i + 1, mask.shape[2]):
                m1 = mask[:, :, j]
                idx = (m1 == 1)
                m00[idx] = 0
            occluded_mask[:, :, i] = m00
        return occluded_mask

    def image_reference(self, image_id):
        """Return the path of the image."""
        if self.read_depth:
            # because of the data collection rule: time_RGB.png for rgb image, time_Depth.png for depth image.
            img_path = self.image_info[image_id]['path'][:-7] + 'Depth.png'
        else:
            img_path = self.image_info[image_id]['path']
        return img_path

    def load_rgb(self, image_id):
        """
        Load the specified rgb image and return a [H,W,3] Numpy array for visualization.
        """
        # Load rgb image        
        img_path = self.image_info[image_id]['path']

        image = skimage.io.imread(img_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image[self.image_info[image_id]['ymin']:self.image_info[image_id]['ymax'],
               self.image_info[image_id]['xmin']:self.image_info[image_id]['xmax']]

    def load_depth(self, image_id):
        """
        Load the specified depth image and return a [H,W,3] Numpy array for visualization.
        """
        # Load rgb image        
        img_path = self.image_info[image_id]['path'][:-7] + 'Depth.png'

        image = skimage.io.imread(img_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image[self.image_info[image_id]['ymin']:self.image_info[image_id]['ymax'],
               self.image_info[image_id]['xmin']:self.image_info[image_id]['xmax']]

    def depth_inpaint(self, depth_img_base, missing_value=0):
        depth_img_base = cv2.copyMakeBorder(depth_img_base, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (depth_img_base == missing_value).astype(np.uint8)
        scale = np.abs(depth_img_base).max()

        depth_img_base = depth_img_base.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        depth_img_base = cv2.inpaint(np.float32(depth_img_base), mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_img_base = depth_img_base[1:-1, 1:-1]
        # depth_img_base = depth_img_base * scale

        return depth_img_base
