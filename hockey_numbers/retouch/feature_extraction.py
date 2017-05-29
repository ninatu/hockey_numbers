#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def get_area(mask):
    return sum(sum(mask > 0))


def get_rect(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return x, y, w, h


def get_centroid(mask):
    moments = cv2.moments(mask)
    x = moments['m10'] / moments['m00']
    y = moments['m01'] / moments['m00']
    return x, y


def normalized_area(mask):
    h, w = mask.shape
    return get_area(mask) / (h * w)


def normalized_rect(mask):
    x, y, w, h = get_rect(mask)
    full_h, full_w = mask.shape
    x = x / full_w
    y = y / full_h
    w = w / full_w
    h = h / full_h
    return x, y, w, h


def normalized_centroid(mask):
    x, y = get_centroid(mask)
    full_h, full_w = mask.shape
    return x / full_w, y / full_h


def get_extent(mask):
    """
    Extent is the ratio of contour area to bounding rectangle area
    """
    area = get_area(mask)
    x, y, w, h = get_rect(mask)
    rect_area = w * h
    extent = float(area) / rect_area
    return extent


def normalized_perimeter(mask):
    mask = mask.copy()
    contour = cv2.findContours(mask, mode=cv2.RETR_LIST , method=cv2.CHAIN_APPROX_NONE)
    perimeter = np.array(contour[0] > 1, dtype=np.uint8).sum()
    h, w = mask.shape
    norm_per = perimeter / (h + w)
    return norm_per


def normalized_orientation(mask):
    mask = mask.copy()
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0, 0, 0, 0, 0
    cnt = np.concatenate(contours, axis=0) 
    if len(cnt) < 5:
        return 0, 0, 0, 0, 0
    (x, y), (w, h), angle = cv2.fitEllipse(cnt)  
    h_mask, w_mask = mask.shape
    return x / w_mask, y / h_mask, w / w_mask, h / h_mask, angle


def mean_color(img, mask):
    pixels = img[np.where(mask)]
    return pixels.mean(axis=0) / 255


def median_color(img, mask):
    pixels = img[np.where(mask)]
    return np.median(pixels, axis=0) / 255


def normalized_centroid_moments(mask):
    moments = cv2.moments(mask)
    return [moments['nu20'], moments['nu02'], moments['nu21'], moments['nu12']]


def create_features(mask):
    fs = list()
    fs.append(normalized_area(mask))
    fs.extend(normalized_rect(mask))
    fs.extend(normalized_centroid(mask))
    fs.append(normalized_perimeter(mask))
    fs.append(get_extent(mask))
    fs.extend(normalized_orientation(mask))
    fs.extend(normalized_centroid_moments(mask))
    return fs


def extract_features(in_X):
    assert len(in_X) != 0
    out_X = np.empty((len(in_X), len(create_features(in_X[0]))))
    for i, obj in enumerate(in_X):
        out_X[i, :] = create_features(obj)
    return out_X