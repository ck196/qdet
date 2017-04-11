#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from settings import  *


def create_gt(image_path, xml_path):
    image = cv2.imread(image_path)
    h, w, c = image.shape

    xscale = IMAGE_SIZE / float(w)
    yscale = IMAGE_SIZE / float(h)

    tree = ET.parse(xml_path)
    objs = tree.findall('object')

    gt_label = np.zeros((CELL, CELL, BOX_PER_CELL, (NUM_CLASSES + 4)))

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = xscale * float(bbox.find('xmin').text) - 1
        y1 = yscale * float(bbox.find('ymin').text) - 1
        x2 = xscale * float(bbox.find('xmax').text) - 1
        y2 = yscale * float(bbox.find('ymax').text) - 1

        box = np.zeros(4)
        box[:] = [x1, y1, x2, y2]

        #box = box / (32 * 7)

        xcenter = int((x2 + x1) / 2)
        ycenter = int((y2 + y1) / 2)

        xbase = xcenter // 32 - 1
        xbl = xcenter % 32

        if xbl > 0:
            xbase += 1

        if xbase == CELL:
            xbase -= 1

        ybase = ycenter // 32 - 1
        ybl = ycenter % 32

        if ybl > 0:
            ybase += 1

        if ybase == CELL:
            ybase -= 1

        cls = CLASS.index(obj.find('name').text.lower().strip())

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        label[cls] = 1.0

        one_label = np.hstack((label, box))

        assign = False
        for idx in range(BOX_PER_CELL):
            if not gt_label[xbase][ybase][idx].any() and not assign:
                gt_label[xbase][ybase][idx] = one_label
                assign = True
                break

        if not assign and xbl == 0:
            ybase += 1
            if ybase == CELL:
                ybase -= 1
            for idx in range(BOX_PER_CELL):
                if not gt_label[xbase][ybase][idx].any() and not assign:
                    gt_label[xbase][ybase][idx] = one_label
                    assign = True
                    break
        if not assign and ybl == 0:
            xbase += 1
            if xbase == CELL:
                xbase -= 1
            for idx in range(BOX_PER_CELL):
                if not gt_label[xbase][ybase][idx].any() and not assign:
                    gt_label[xbase][ybase][idx] = one_label
                    break

    return gt_label.flatten(), image


# a, b = create_gt("data/2007_001185.jpg", "data/2007_001185.xml")
# print a.shape, b.shape


# Direction = 0 -> vertically
# Direction = 1 -> hozirontally
def flip(image, direction=0):
    flip_img = image.copy()
    if direction == 0:
        flip_img = cv2.flip(flip_img, 0)
    else:
        flip_img = cv2.flip(flip_img, 1)
    return flip_img

