#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Remove no players images from hdf5 dataset
Uses externals classifier"""

import argparse
import h5py
import tqdm
import sys
import cv2
import numpy as np
from sklearn.externals import joblib
import os.path as osp

CLASSIFY_PATH = './classifying'
CLASSIFY_DATA_PATH = '/media/nina/Seagate Backup Plus Drive/hockey/other/samples/'
CLASSIFIER_H, CLASSIFIER_W = (64, 32)

sys.path.append(osp.join(osp.dirname(__file__), CLASSIFY_PATH))
print(osp.join(osp.dirname(__file__), CLASSIFY_PATH))
from classifying.classify import get_classifier, extract_feature

def remove_not_players(infile, outfile):
    """Remove no players images from hdf5 dataset
    
    Keyword arguments:
    infile - hdf5 file with group image and mask
    outfile - hdf5 file to save result
    """

    in_db =  h5py.File(infile)
    out_db = h5py.File(outfile, 'w')
    out_db.create_group('/image')
    out_db.create_group('/mask')
    imnames = sorted(in_db['image'].keys())

    #load classifier
    clf = get_classifier(CLASSIFY_DATA_PATH, None, None)
    buckets = joblib.load(CLASSIFY_DATA_PATH + 'buckets.pkl')
    
    for imname in tqdm.tqdm(imnames):
        image = in_db['image'][imname].value
        mask = in_db['mask'][imname].value

        image = image.astype(np.float32) / 255.0
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        data = np.concatenate((image, mask), axis=2)
        data = cv2.resize(data, (CLASSIFIER_H, CLASSIFIER_W))
        features = extract_feature(data, buckets)
        team = clf.predict(features)
        if team == 1 or team == 2:
            out_db['image'].copy(in_db['image'][imname], imname)
            out_db['mask'].copy(in_db['mask'][imname], imname)

def main():
    """Parse args and run function"""

    parser = argparse.ArgumentParser(description="Remove no players images from hdf5 dataset")
    parser.add_argument('infile', type=str, nargs=1, help="hdf5 file with group image and mask")
    parser.add_argument('outfile', type=str, nargs=1, help="outfile - hdf5 file to save result")
    args = parser.parse_args()
    remove_not_players(args.infile[0], args.outfile[0])

if __name__ == '__main__':
   main()
