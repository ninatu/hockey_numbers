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
parser.add_argument('mark', type=str, nargs='?', help='image mark to be saved')

frame_dir = '/media/nina/Seagate Backup Plus Drive/hockey/frames'
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

for name, objs in marking.getByMark(args.mark).items():
    frame = cv2.imread(os.path.join(frame_dir, name))
    for obj in objs:
        rect = obj['body_rect']
        img = cropByRect(frame, rect)
        img = np.array(img)
        outpath = os.path.join(outdir, '{}.png'.format(md5_hash(img)))
        cv2.imwrite(outpath, img)








