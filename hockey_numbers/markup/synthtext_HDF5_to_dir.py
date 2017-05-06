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

def md5_hash(arr):
    md5 = hashlib.md5() 
    arr = base64.b64encode(arr)
    md5.update(arr)
    return md5.hexdigest()

infile = h5py.File(args.h5file, 'r')
dataGroup = infile['data']
outdir = args.outdir

if not os.path.exists(outdir):
    os.mkdir(outdir)

for n in range(1, 100):
    dirname = '{}'.format(n)
    dirname = os.path.join(outdir, dirname)
    if not os.path.exists(dirname):
        os.mkdir(dirname)


for name, img_data in tqdm.tqdm(dataGroup.items()):
    #mark = img_data.attrs['txt'].decode('utf-8')
    mark = img_data.attrs['txt'][0].decode('utf-8')
    if int(mark) > 99:
        continue
    img = img_data.value
    dirname = os.path.join(outdir, mark)
    outpath = os.path.join(dirname, '{}.png'.format(md5_hash(img)))
    img = Image.fromarray(img)
    img.save(outpath)









