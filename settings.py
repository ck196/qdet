#!/usr/bin/env python
# -*- coding: utf-8 -*-


CLASS = ('__background__', # always index 0
         'aeroplane', 'bicycle', 'bird', 'boat',
         'bottle', 'bus', 'car', 'cat', 'chair',
         'cow', 'diningtable', 'dog', 'horse',
         'motorbike', 'person', 'pottedplant',
         'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = 21
BOX_PER_CELL = 2
CELL = 7
IMAGE_SIZE = 224
LABEL_SHAPE = CELL * CELL * (NUM_CLASSES + 4) * BOX_PER_CELL

BATCH_SIZE = 5

IMAGE_DIR = "/data/workspace/pva-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages"
ANNO_DIR = "/data/workspace/pva-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations"