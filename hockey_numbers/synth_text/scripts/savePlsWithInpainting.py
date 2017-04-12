# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../number_recognition"))

from utils.segmentation import colorSegByMask
from utils.marking import Marking
from utils.imageProcess import bgr2rgb, cropByRect
from inpaintingNumbers import inpainting, classify

import random
import cv2
import re
import h5py
import argparse
import numpy as np

DEFAULT_FRAME_DIR = '/media/nina/Seagate Backup Plus Drive/hockey/rowNumbers/imgWithMask'
DEFAULT_MAX_COUNT = int(1e9)


parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', '--outputFile', required=True, type=str, nargs='?', help='file to save result hdf5 file')
parser.add_argument('-n ', '--maxCount', default=DEFAULT_MAX_COUNT, type=int, nargs='?', help='max count of saving croped players')
parser.add_argument('--imageAndMaskDir', default=DEFAULT_FRAME_DIR,  type=str, nargs='?', help='dir with video frames and their masks')


args = parser.parse_args()
MAX_COUNT = args.maxCount
dsetOutput = args.outputFile
imageDir = args.imageAndMaskDir 

templateImg = '{:d}.png'
templateMask = 'mask{:d}.png'

imgNumbers = map(lambda x: int(re.findall(r'\d+', x)[0]), os.listdir(imageDir))
imgNumbers = list(set(map(int, imgNumbers)))

fout = h5py.File(dsetOutput, 'w')
imageGr = fout.create_group('image')

i = 0
clf = classify.getClassifier()
for imNumb in imgNumbers:
    if i >= MAX_COUNT:
        break
    img = cv2.imread(os.path.join(imageDir, templateImg.format(imNumb)))
    mask = cv2.imread(os.path.join(imageDir, templateMask.format(imNumb)))
    mask = np.array(mask[:,:,0], dtype=np.uint8)

    numberMasks= classify.findMaskOfNumber(img, mask, clf)
    if (len(numberMasks) < 2):
        continue

    img = bgr2rgb(img)
    newImg = img.copy()    
    for m in map(lambda x: x[0], numberMasks):
        m = inpainting.colorAndDistDilateMask(inImg=img, mask=m, colorCoeff=4, distStep=4)
        newImg = inpainting.inpainting2(inImg=newImg, mask=m)
    
    nameImg = templateImg.format(i)
    imageGr.create_dataset(nameImg, newImg.shape, '|u1', newImg)
    i += 1
fout.close()
print("Save to database {:d} image".format(i))    

