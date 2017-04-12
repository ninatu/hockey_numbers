#!/usr/bin/python3

import argparse
import os
import sys
import hashlib
import base64
import cv2
import numpy as np
from libs.marking import Marking
from libs.imageProcess import cropByRect


parser = argparse.ArgumentParser()
parser.add_argument('marking_file', type=str, nargs='?', help='input marking file')
parser.add_argument('outdir', type=str, nargs='?', help='dir to ouput')

frame_dir = '/media/nina/Seagate Backup Plus Drive/hockey/frames'
masks_dir = '/media/nina/Seagate Backup Plus Drive/hockey/masks'

args = parser.parse_args()

def md5_hash(arr):
    md5 = hashlib.md5() 
    arr = base64.b64encode(arr)
    md5.update(arr)
    return md5.hexdigest()

marking = Marking()
with open(args.marking_file) as fin:
    marking.addFromJson(fin)
outdir = args.outdir

if not os.path.exists(outdir):
    os.mkdir(outdir)

for n in range(1, 100):
    dirname = '{}'.format(n)
    dirname = os.path.join(outdir, dirname)
    if not os.path.exists(dirname):
        os.mkdir(dirname)


for name, objs  in marking.getByMark('number').items():
    frame = cv2.imread(os.path.join(frame_dir, name))
    for obj in objs:
        mark = obj['number']
        rect = obj['body_rect']
        img = cropByRect(frame, rect)
        img = np.array(img)
        dirname = os.path.join(outdir, mark)
        outpath = os.path.join(dirname, '{}.png'.format(md5_hash(img)))
        cv2.imwrite(outpath, img)








