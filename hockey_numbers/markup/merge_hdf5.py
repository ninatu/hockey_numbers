import argparse
import h5py
import os.path as osp
import numpy as np
import tqdm
from utils.image_process import md5_hash
from constants import TEMPLATE_IMAGE

parser = argparse.ArgumentParser(description='')
parser.add_argument('infiles', nargs='*', type=str, help='hdf5s')
parser.add_argument('-o', '--outfile', required=True, type=str, help='file to save results')
parser.add_argument('-p', '--path', required=True, type=str, help='path in database fir merge')

args = parser.parse_args()
infiles = args.infiles
outfile = args.outfile
path = args.path

out_db = h5py.File(outfile, 'w')
out_db.create_group(path)

n = 0
for infile in tqdm.tqdm(infiles):
    in_db = h5py.File(infile)
    for img_name in tqdm.tqdm(in_db[path].keys()):
        img = in_db[path][img_name].value
        out_name = TEMPLATE_IMAGE.format(md5_hash(img))
        out_db[path].copy(in_db[path][img_name], out_name)
        n += 1
    in_db.close()
out_db.close()
print("COUNT SAVED IMAGES: {}".format(n))

