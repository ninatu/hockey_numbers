import argparse
import h5py
import os.path as osp
import numpy as np
import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('infiles', nargs='*', type=str, help='hdf5s with images')
parser.add_argument('-o', '--outfile', required=True, type=str, help='file to save results')

args = parser.parse_args()
infiles = args.infiles
outfile = args.outfile

outDB = h5py.File(outfile, 'w')
outImgGr = outDB.create_group('image')

templateImage = '{:d}.png'
n = 0
for infile in tqdm.tqdm(infiles):
    inDB = h5py.File(infile)
    inImgGr = inDB['image']
    imgKeys = sorted(list(inImgGr.keys()))
    for imgName in tqdm.tqdm(imgKeys):
        img = inImgGr[imgName].value
        outName = templateImage.format(n)
        _imset = outImgGr.create_dataset(outName, img.shape, '|u1', img)
         #saving attrs
        imgAttrs = inImgGr[imgName].attrs
        for key in imgAttrs.keys():
            _imset.attrs[key] = imgAttrs[key]
        n += 1
    inDB.close()
outDB.close()
print("COUNT SAVED IMAGES: {}".format(n))

