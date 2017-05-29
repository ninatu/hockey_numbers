#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import h5py
import tqdm
from classify import create_classifier, get_classifier
from number_retouching import retouch_number


def train(args):
    create_classifier(args.dir, args.output)


def retouch(args):
    in_db = h5py.File(args.input, 'r')
    out_db = h5py.File(args.output, 'w')
    out_db.create_group('image')

    clf = get_classifier()

    for img_name in tqdm.tqdm(in_db['image'].keys()):
        img = in_db['image'][img_name].value
        mask = in_db['mask'][img_name].value
        retouch_img = retouch_number(img, mask, clf)

        if retouch_img is not None:
            img_dset = out_db['image'].create_dataset(img_name, retouch_img.shape, '|u1', retouch_img)
            for key in in_db['image'][img_name].attrs.keys():
                img_dset.attrs[key] = in_db['image'][img_name].attrs[key]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Retouching number on image')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='train segment classifier')
    train_parser.add_argument('dir', type=str, help='directory with marked segmentation image')
    train_parser.add_argument('-o', '--output', type=str, required=True, help='path to save trained classifier')
    train_parser.set_defaults(func=train)

    retouch_parser = subparsers.add_parser('retouch', help='retouch numbers')
    retouch_parser.add_argument('input', type=str, help='hdf5 file with masks and images')
    retouch_parser.add_argument('-o', '--output', type=str, required=True, help='path to save retouching images')
    retouch_parser.set_defaults(func=retouch)

    args = parser.parse_args()
    args.func(args)
