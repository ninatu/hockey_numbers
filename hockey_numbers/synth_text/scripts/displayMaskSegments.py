# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io
import random

mcg = scipy.io.loadmat('/home/nina/Documents/hockey_tracking/number_recognition/SynthText/mcg/pre-trained/mcgrouping.mat')['mcgrouping']
masks = mcg['masks'][0][0]
scores = mcg['scores']['0']

countSegment = 50
segmentMap = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)

for i in range(countSegment):
    curMask = masks[:,:,i]
    curColor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    segmentMap[np.where(curMask > 0)] = curColor
    plt.figure()
    plt.imshow(segmentMap)
    plt.show()
    

