#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../number_recognition"))

from utils.segmentation import colorSegByMask
from utils.marking import Marking
from utils.imageProcess import bgr2rgb, cropByRect
from inpaintingNumbers import inpainting, classify

import json
import numpy as np
import random
import cv2
import re
import h5py
import argparse
import tqdm

DEFAULT_FRAME_DIR = '/media/nina/Seagate Backup Plus Drive/hockey/frames/'
DEFAULT_MASK_DIR = '/media/nina/Seagate Backup Plus Drive/hockey/masks/'
DEFAULT_MAX_COUNT = 1e9


parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--markingFile', required=True, type=str, nargs='?', help='file with marking of players')
parser.add_argument('-o', '--outputFile', required=True, type=str, nargs='?', help='file to save result hdf5 file')
parser.add_argument('-n ', '--maxCount', default=DEFAULT_MAX_COUNT, type=int, nargs='?', help='max count of saving croped players')
parser.add_argument('--inpaint', action='store_true', help='get images with number and inpaint number')
parser.add_argument('--imageDir', default=DEFAULT_FRAME_DIR,  type=str, nargs='?', help='dir with video frames')
parser.add_argument('--maskDir', default=DEFAULT_MASK_DIR,  type=str, nargs='?', help='dir with masks of video frames')
parser.add_argument('--frameNumbers', default=None, type=str, nargs='?', help='json file with list number frame to proceed')


args = parser.parse_args()
MAX_COUNT = args.maxCount
dsetOutput = args.outputFile
markingPath = args.markingFile
imageDir = args.imageDir
maskDir = args.maskDir
inpaint = args.inpaint
pathFrameNumbers = args.frameNumbers

templateImage = '{:d}.png'
templateMask = 'mask{:d}.png'
patNumb = re.compile(r'\d+')
def getNumber(name):
    return int(patNumb.search(name).group(0))

marking = Marking()
marking.addFromJson(open(markingPath))
fout = h5py.File(dsetOutput, 'w')
imageGr = fout.create_group('image')
if pathFrameNumbers:
    with open(pathFrameNumbers) as fin:
        frameNumbers = json.load(fin)
else:
    frameNumbers = None

if inpaint:
    imageDict = marking.getByMark("number")
    imageNames = sorted(list(imageDict.keys()))
    if frameNumbers:
        imageNames = set(imageNames).intersection(set(frameNumbers))
    i = 0
    clf = classify.getClassifier()
    for name in tqdm.tqdm(imageNames):    
        frame = cv2.imread(os.path.join(imageDir, name))
        maskname = templateMask.format(getNumber(name))
        maskframe = cv2.imread(os.path.join(maskDir, maskname))
        maskframe = np.array(maskframe[:,:,0], dtype=np.uint8)

        for obj in imageDict[name]:
            body_rect = obj['body_rect']
            img = cropByRect(frame, body_rect)
            mask = cropByRect(maskframe, body_rect)
            numberMasks= classify.findMaskOfNumber(img, mask, clf)
            if (len(numberMasks) < 2):
                continue
            img = bgr2rgb(img)
            newImg = img.copy()    
            for m in map(lambda x: x[0], numberMasks):
                m = inpainting.colorAndDistDilateMask(inImg=img, mask=m, colorCoeff=4, distStep=4)
                newImg = inpainting.inpainting2(inImg=newImg, mask=m)
            img = newImg
            nameImg = templateImage.format(i)
            imgDset = imageGr.create_dataset(nameImg, img.shape, '|u1', img)
            imgDset.attrs['source_name'] = name
            for key in body_rect.keys():
                imgDset.attrs[key] = body_rect[key]
            
            i += 1
            if i >= MAX_COUNT:
                break
        if i >= MAX_COUNT:
            break
else:
    imageDict = marking.getByMark("number_isnt_visible")
    imageNames = sorted(list(imageDict.keys()))
    if frameNumbers:
        imageNames = set(imageNames).intersection(set(frameNumbers))
    i = 0
    for name in tqdm.tqdm(imageNames):    
        frame = cv2.imread(os.path.join(imageDir, name))

        for obj in imageDict[name]:
            body_rect = obj['body_rect']
            img = cropByRect(frame, body_rect)
            img = bgr2rgb(img)
            nameImg = templateImage.format(i)
            imgDset = imageGr.create_dataset(nameImg, img.shape, '|u1', img)
            imgDset.attrs['source_name'] = name
            for key in body_rect.keys():
                imgDset.attrs[key] = body_rect[key]
            i += 1
            if i >= MAX_COUNT:
                break
        if i >= MAX_COUNT:
            break

print("COUNT SAVED IMAGES: {}".format(i))
fout.close()

