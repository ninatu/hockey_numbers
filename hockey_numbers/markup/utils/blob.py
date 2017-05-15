#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Work with blobs on images"""

import cv2

from constants import BLOB_MIN_HEIGHT, BLOB_MAX_HEIGHT, BLOB_MIN_WIDTH, BLOB_MAX_WIDTH
from constants import FIELD_Y, FIELD_H, FIELD_X, FIELD_W


class Blob(object):
    def __init__(self, x, y, width, height, area, centroid, mask=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = area
        # centoid is a numpy array
        self.centroid = centroid.copy()
        self.mask = mask

    @staticmethod
    def create_from_stats(stats, centroid, mask=None):
        return Blob(x=stats[cv2.CC_STAT_LEFT],
                    y=stats[cv2.CC_STAT_TOP],
                    width=stats[cv2.CC_STAT_WIDTH],
                    height=stats[cv2.CC_STAT_HEIGHT],
                    area=stats[cv2.CC_STAT_AREA],
                    centroid=centroid,
                    mask=mask)


def blobs_IOU(blob1, blob2):
    #intersection

    """right intersection
    mask1 = np.reshape(blob1.mask, (blob1.mask.shape[0], blob1.mask.shape[1], 1)).astype(np.int)
    mask2 = np.reshape(blob2.mask, (blob2.mask.shape[0], blob2.mask.shape[1], 1)).astype(np.int)
    intersect = sum(sum(cv2.bitwise_and(mask1, mask2)))
    """
    # box intersection
    dx = min(blob1.x + blob1.width, blob2.x + blob2.width) - max(blob1.x, blob2.x)
    dy = min(blob1.y + blob1.height, blob2.y + blob2.height) - max(blob1.y, blob2.y)
    if (dx >= 0) and (dy >= 0):
        intersect = dx * dy
        union = blob1.width * blob1.height + blob2.width * blob2.height - intersect
        return float(intersect) / union
    else:
        return 0.0


def get_nearest_blob(blob, blobs_list, min_IOU=0.2):
    max_IOU = 0
    best_i = -1
    for i, candidate in enumerate(blobs_list):
        iou = blobs_IOU(blob, candidate)
        if iou > max_IOU:
            max_IOU = iou
            best_i = i
    if max_IOU > min_IOU:
        return best_i
    else:
        return -1


def filter_blobs_by_size(blobs, min_height=BLOB_MIN_HEIGHT,
                         max_height=BLOB_MAX_HEIGHT,
                         min_width= BLOB_MIN_WIDTH,
                         man_width=BLOB_MAX_WIDTH):
    good_blobs = []
    for blob in blobs:
        if min_height <= blob.height <= max_height and min_width <= blob.width <= man_width:
            good_blobs.append(blob)
    return good_blobs


def filter_blobs_by_field(blobs, y=FIELD_Y, h=FIELD_H, x=FIELD_X, w=FIELD_W):
    good_blobs = []
    for blob in blobs:
        if y <= blob.y and blob.y + blob.height <= h and \
           x <= blob.x and blob.x + blob.width <= w:
            good_blobs.append(blob)

    return good_blobs


def get_blobs_from_masks(mask, save_masks=False):
    ret_val, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if save_masks:
        blobs = [Blob.create_from_stats(stats[i, :], centroids[i, :], labels==i) for i in range(stats.shape[0])]
    else:
        blobs = [Blob.create_from_stats(stats[i, :], centroids[i, :]) for i in range(stats.shape[0])]
    return blobs
