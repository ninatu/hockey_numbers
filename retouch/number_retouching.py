# -*- coding: utf-8 -*-

import numpy as np
import cv2
from segmentation import color_segmentation, get_masks_of_segments
from feature_extraction import extract_features
from retouching import retouch, color_and_dist_dilate_mask


def find_number_mask(img, mask, clf):
    """
    return None, if error in finding number mask
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    color_seg = color_segmentation(img, mask=mask, n_colors=4)
    masks_of_seg = get_masks_of_segments(color_seg)
    X = extract_features(masks_of_seg)
    y = clf.predict(X)
    if y.sum() < 2:
        return None

    mask_of_number = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i, mask in enumerate(masks_of_seg):
        if y[i]:
            mask_of_number = np.logical_or(mask, mask_of_number)
    return mask_of_number.astype(np.uint8)


def retouch_number(img, mask, clf):
    mask_of_number = find_number_mask(img, mask, clf)
    if mask_of_number is None:
        return None
    mask_of_number = color_and_dist_dilate_mask(img.copy(), mask_of_number, color_coeff=4, dist_step=4)
    retouch_img = retouch(img.copy(), mask_of_number)
    return retouch_img