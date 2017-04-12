#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Work with blobs on images"""

import cv2
import numpy as np

from hockey_numbers.markup.constants import BLOB_MIN_HEIGHT, BLOB_MAX_HEIGHT, BLOB_MIN_WIDTH, BLOB_MAX_WIDTH
from hockey_numbers.markup.constants import FIELD_Y, FIELD_H, FIELD_X, FIELD_W

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

def blobsIOU(blob1, blob2):
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


def getNearestBlob(blob, listBlobs, minIOU=0.2):
    maxIOU = 0
    bestI = -1;
    for i, candidat in enumerate(listBlobs):
        iou = blobsIOU(blob, candidat)
        if iou > maxIOU:
            maxIOU = iou
            bestI = i
    if maxIOU > minIOU:
        return bestI
    else:
        return -1


def filterBlobsBySize(blobs, minHeight=BLOB_MIN_HEIGHT,
                      maxHeight=BLOB_MAX_HEIGHT,
                      minWidth= BLOB_MIN_WIDTH,
                      manWidth=BLOB_MAX_WIDTH):
    goodBlobs = []
    for blob in blobs:
        if minHeight <= blob.height <= maxHeight and \
                                minWidth <= blob.width <= manWidth:
            goodBlobs.append(blob)
    return goodBlobs

def filterBlobsByField(blobs, minHeight=FIELD_Y,
                       maxHeight=FIELD_H,
                       minWidth=FIELD_X,
                       manWidth=FIELD_W):
    goodBlobs = []
    for blob in blobs:
        if minHeight <= blob.y and blob.y  + blob.height <= maxHeight and \
            minWidth <= blob.x and blob.x + blob.width <= manWidth:

            goodBlobs.append(blob)
    return goodBlobs


def getBlobsFromMasks(mask, saveMasks=False):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if saveMasks:
        blobs = [Blob.create_from_stats(stats[i, :], centroids[i, :], labels==i) for i in range(stats.shape[0])]
    else:
        blobs = [Blob.create_from_stats(stats[i, :], centroids[i, :]) for i in range(stats.shape[0])]
    return blobs
