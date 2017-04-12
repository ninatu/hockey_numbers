# -*- coding: utf-8 -*-
import numpy as np
import cv2


def segmentNormalize(inputSegmentImg):
    #uniq label segment from rgb
    badSegmentMarking = inputSegmentImg[:,:,0] + \
                      (inputSegmentImg[:,:,1] * 256) + \
                      (inputSegmentImg[:,:,2] * 256 * 256)
    uniqLabels = np.unique(badSegmentMarking)
    
    # numbering segment, count area
    labels = []
    areas = []
    goodSegmentMarking = np.empty(badSegmentMarking.shape, dtype=np.uint8)
    
    for newLabel, oldLabel in enumerate(uniqLabels):
        segmentMask = (badSegmentMarking == oldLabel)
        goodSegmentMarking[np.where(segmentMask)] = newLabel
        labels.append(newLabel)
        areas.append(np.count_nonzero(segmentMask))
        
    return goodSegmentMarking, labels, areas
    

def segmentFromContourMap(contourMap):
    """
    contourMap - hxw binary array, when value greater than 0 - contour
    """
    segmentMap = np.array(contourMap == 0, np.uint8)
    (nLabels, seg) = cv2.connectedComponents(segmentMap)
    labels = range(1, nLabels)
    areas = [(seg == label).sum() for label in labels]
    return seg, labels, areas
    

if __name__ == '__main__':
    #filename = '/home/nina/Documents/hockey_tracking/number_recognition/number_recognition/data/playersWithoutNumber/424628.png'
    #inputSegmentImg = cv2.imread(filename)
    #seg, labels, areas = segmentFromContourMap(inputSegmentImg[:, :, 0] / 256 > 0.11)
    #segmentMarking, label, area = segmentNormalize(inputSegmentImg)
    import scipy.io
    contourMap = scipy.io.loadmat('/home/nina/Documents/hockey_tracking/number_recognition/SynthText/data/img2/ucm.mat')['ucm']
    