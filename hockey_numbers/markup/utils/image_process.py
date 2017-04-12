#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def getOrientation(bin_image, union_all=False):
    _, contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if union_all:
        cnt = np.concatenate(contours, axis=0)
    else:
        cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    center, size, angle = ellipse
    if angle > 90:
        angle = angle - 180
    """
    if -45 <= angle < 45:
        return center, size, angle
    elif 45 <= angle < 135:
        angle = -90 + angle
        size = size[1], size[0]
    elif 135 <= angle:
        angle = -180 + angle
    elif -135 <= angle < -45:
        angle = angle + 90
        size = size[1], size[0]
    elif angle < -135:
        angle = 180 + angle
    """
    return center, size, angle


def getBoundingBox(bin_image, union_all=False):
    _, contours, hierarchy = cv2.findContours((bin_image * 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if union_all:
        cnt = np.concatenate(contours, axis=0)
    else:
        cnt = contours[0]
    return cv2.boundingRect(cnt)


def scaleImage(image, scale):
    rot_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2.0, image.shape[0] / 2.0), 0, scale)
    rows = image.shape[0]
    cols = image.shape[1]
    image = cv2.warpAffine(image, rot_matrix, (cols, rows))
    return image


def cropByMask(image, mask, center):
    n_x, n_y, n_w, n_h = getBoundingBox(mask, True)
    mask = mask[n_y:n_y + n_h, :][:, n_x:n_x + n_w]
    image = image[n_y:n_y + n_h, :][:, n_x:n_x + n_w]
    center = center[0] - n_x, center[1] - n_y
    return image, mask, center, (n_w, n_h)

def cropByRect(frame, rect):
    x, y, w, h = map(int, [rect["x"], rect["y"], rect["w"],rect["h"]])
    return frame[y:y+h, x:x+w]

def bgr2lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

def lab2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr2hsv(img):
    return cv2.cvtColor(img,  cv2.COLOR_BGR2HSV)

def hsv2rgb(img):
    return cv2.cvtColor(img,  cv2.COLOR_HSV2RGB)
    
def hsv2bgr(img):
    return cv2.cvtColor(img,  cv2.COLOR_HSV2BGR)

def morphologyClosing(binaryImg):
    return cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, np.ones((3, 3)))



