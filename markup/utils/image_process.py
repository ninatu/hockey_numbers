#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import hashlib
import base64


def get_orientation(bin_image, union_all=False):
    _, contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if union_all:
        cnt = np.concatenate(contours, axis=0)
    else:
        cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    center, size, angle = ellipse
    if angle > 90:
        angle = angle - 180
    return center, size, angle


def get_bounding_box(bin_image, union_all=False):
    _, contours, hierarchy = cv2.findContours((bin_image * 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if union_all:
        cnt = np.concatenate(contours, axis=0)
    else:
        cnt = contours[0]
    return cv2.boundingRect(cnt)


def scale_image(image, scale):
    rot_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2.0, image.shape[0] / 2.0), 0, scale)
    rows = image.shape[0]
    cols = image.shape[1]
    image = cv2.warpAffine(image, rot_matrix, (cols, rows))
    return image


def crop_by_mask(image, mask, center):
    n_x, n_y, n_w, n_h = get_bounding_box(mask, True)
    mask = mask[n_y:n_y + n_h, :][:, n_x:n_x + n_w]
    image = image[n_y:n_y + n_h, :][:, n_x:n_x + n_w]
    center = center[0] - n_x, center[1] - n_y
    return image, mask, center, (n_w, n_h)


def crop_by_rect(frame, x, y, w, h):
    return frame[y:y+h, x:x+w]


def md5_hash(arr):
    arr = np.array(arr, dtype=arr.dtype)
    md5 = hashlib.md5()
    arr = base64.b64encode(arr)
    md5.update(arr)
    return md5.hexdigest()


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


def morphology_closing(binaryImg):
    return cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, np.ones((3, 3)))
