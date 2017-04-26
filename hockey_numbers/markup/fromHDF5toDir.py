#!/usr/bin/python3

import h5py
import argparse
import os
import sys
import hashlib
import base64
import tqdm
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('h5file', type=str, nargs='?', help='input h5py file with images')
parser.add_argument('outdir', type=str, nargs='?', help='dir to ouput')

args = parser.parse_args()

infile = h5py.File(args.h5file, 'r')
dataGroup = infile['image']
outdir = args.outdir

if not os.path.exists(outdir):
    os.mkdir(outdir)


for name, img_data in tqdm.tqdm(dataGroup.items()):
    outpath = os.path.join(dirname, name)
    img = Image.fromarray(img)
    img.save(outpath)









