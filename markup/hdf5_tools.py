#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.image_process import md5_hash
from constants import TEMPLATE_IMAGE

import h5py
import argparse
import os
import sys
import hashlib
import base64
import tqdm
from PIL import Image


def merge_hdf5(args):
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


def save_from_hdf5(args):
    infile = args.infile
    outdir = args.outdir
    path = args.path

    in_db = h5py.File(infile, 'r')
    data_group = in_db[path]

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for name, img_data in tqdm.tqdm(data_group.items()):
        out_path = os.path.join(outdir, name)
        img = Image.fromarray(img)
        img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_merge = subparsers.add_parser('merge', help='merge hfd5 files by path')
    parser_merge.add_argument('infiles', nargs='*', type=str, help='hdf5s')
    parser_merge.add_argument('-o', '--outfile', required=True, type=str, help='file to save results')
    parser_merge.add_argument('-p', '--path', required=True, type=str, help='path in database for merge')
    parser_merge.set_defaults(func=merge_hdf5)

    parser_save = subparsers.add_parser('save', help='save images from hfd5 file to dir')
    parser_save.add_argument('infile', type=str, nargs='?', help='input h5py file with images')
    parser_save.add_argument('outdir', type=str, nargs='?', help='dir to ouput')
    parser_save.add_argument('-p', '--path', required=True, type=str, help='path in database from save')
    parser_save.set_defaults(func=save_from_hdf5)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()