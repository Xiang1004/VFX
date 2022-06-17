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
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import edge_model as modellib
from mrcnn import utils
from myData import TargetsConfig, TargetsDataset

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def train(model, args):
    """Train the model."""
    aug = iaa.Sometimes(0.5, [
        iaa.Crop(percent=(0, 0.3)),
        iaa.Fliplr(0.8),
        # iaa.Flipud(0.8),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.LinearContrast((0.75, 2.5)),
        iaa.Add((-15, 15), per_channel=0.5),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ),
        iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255)),
        iaa.Superpixels(p_replace=(0, 0.1), n_segments=(20, 200)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
        iaa.EdgeDetect(alpha=(0.5, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise to images
        iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
        iaa.Invert(0.05, per_channel=True),  # invert color channels
        # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.FrequencyNoiseAlpha(
            exponent=(-4, 0),
            first=iaa.Multiply((0.5, 1.5), per_channel=True),
            second=iaa.LinearContrast((0.5, 2.0))
        ),
        # iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),  # move pixels locally around (with random strengths)
        iaa.PerspectiveTransform(scale=(0.01, 0.1))

    ])

    YAML_DIR = args.yaml
    print('read yaml file from', YAML_DIR)

    with open(YAML_DIR, 'r') as stream:
        data = yaml.full_load(stream)

    dataset_train = TargetsDataset(train_split=None, read_depth=args.depth_only, input_RGBD=args.RGBD)

    #   for h in data:
    for h in ['train']:
        print('reading training data', h, '......')
        Targets_DIR = data[h]['path']
        print('Targets_DIR', Targets_DIR)
        CROP = data[h]['crop']
        dataset_train.load_targets(Targets_DIR, CROP['xmin'], CROP['xmax'], CROP['ymin'], CROP['ymax'])

    # Must call before using the dataset
    dataset_train.prepare()
    dataset_val = TargetsDataset(train_split=None, val_dataset=False, read_depth=args.depth_only, input_RGBD=args.RGBD)

    #   for h in data:
    for h in ['val']:
        print('reading testing data', h, '......')
        Targets_DIR = data[h]['path']
        CROP = data[h]['crop']
        dataset_val.load_targets(Targets_DIR, CROP['xmin'], CROP['xmax'], CROP['ymin'], CROP['ymax'])

    # Must call before using the dataset
    dataset_val.prepare()

    print("Image Count | train:{} val:{}".format(len(dataset_train.image_ids), len(dataset_val.image_ids)))
    print("Class Count:{}".format(dataset_train.num_classes))

    test_image = dataset_train.load_image(0)
    print("Data Shape: {}".format(test_image.shape))

    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.epochs,
                layers='heads',
                augmentation=aug
                )


############################################################
#  Training
############################################################
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--gray', action='store_true', default=False, help='turn RGB to gray scale input')
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--yaml', required=False, default=None, help='datasets information yaml path')
    parser.add_argument('--RGBD', action='store_true', default=True, help='Training on RGB and depth data')
    parser.add_argument('--depth_only', action='store_true', default=False, help='Training on depth data only')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')

    args = parser.parse_args()
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.RGBD:
        print('RGBD!!!')
        TargetsConfig.IMAGE_CHANNEL_COUNT = 4
        TargetsConfig.MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 0.9683])
    elif args.depth_only:
        TargetsConfig.MEAN_PIXEL = np.array([0.9683, 0.9683, 0.9683])

    config = TargetsConfig()
    config.display()

    # Create model    
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "conv1"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train 
    train(model, args)
