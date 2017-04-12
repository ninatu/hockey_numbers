# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import cv2
import scipy.io
import numpy as np
import h5py
from segmentation import segmentNormalize, segmentFromContourMap
import random

def createDataset():
    imgFolder = '/home/nina/Documents/hockey_tracking/number_recognition/SynthText/data/img2/'
    dsetFolder = '/home/nina/Documents/hockey_tracking/number_recognition/SynthText/data'
    N=50   
    img = cv2.imread(os.path.join(imgFolder, 'image.png'))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = scipy.io.loadmat(os.path.join(imgFolder, 'predict_depth.mat'))['data_obj']
    depth = depth.T
    depth = depth.reshape(1, depth.shape[0], depth.shape[1])
    depth = np.concatenate((depth, depth), axis = 0)
    ucm = scipy.io.loadmat(os.path.join(imgFolder, 'ucm.mat'))['ucm']
    seg, label, area = segmentFromContourMap(ucm > 0.30)
    
    dset = h5py.File(os.path.join(dsetFolder, 'myDset2.h5'), 'w')
    imageGr = dset.create_group('image')
    segGr = dset.create_group('seg')
    depthGr = dset.create_group('depth')    
    
    for i in range(N):
        nameImg = 'img{}.jpg'.format(i)
        imageGr.create_dataset(nameImg, img.shape, '|u1', img)
        depthGr.create_dataset(nameImg, depth.shape, '<f4', depth)
        segImg = segGr.create_dataset(nameImg, seg.shape, "<u2", seg)
        segImg.attrs['label'] = label
        segImg.attrs['area'] = area        
    dset.close()
    
def visualizeSynthText(filename):
    dset = h5py.File(filename, 'r')
    images = dset['data']
    imgNames = list(images)
    random.shuffle(imgNames)
    N_row = 10
    plt.figure(figsize= (20, 30))
    for i, key in enumerate(imgNames):
        if i >= N_row ** 2:
            break
        img = images[key]
        plt.subplot(N_row, N_row, i + 1)
        plt.imshow(img)
        plt.axis('off')
    #plt.show()
    plt.savefig('/home/nina/Documents/hockey_tracking/number_recognition/SynthText/data/to report/result/plotNumbers22.png')
    dset.close()
    
    
    
if __name__ == '__main__':
    filename = '/home/nina/Documents/hockey_tracking/number_recognition/SynthText/data/synthtext/largest_bad10'
    visualizeSynthText(filename)
    
    

    