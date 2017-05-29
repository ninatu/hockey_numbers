#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare data for markup"""

from utils.markup import Markup
from constants import TEMPLATE_FRAME
from save_frames_to_dir import save_frames_to_dir

import tqdm
import h5py
import argparse
import os
import shutil


def save_markup(hdf5_path, out_path, hdf5_image_path='image'):
    in_db = h5py.File(hdf5_path)
    markup = Markup()

    for img_name, img_dset in tqdm.tqdm(in_db[hdf5_image_path].items()):
        frame_numb = img_dset.attrs['frame_number']
        x = img_dset.attrs['x']
        y = img_dset.attrs['y']
        w = img_dset.attrs['w']
        h = img_dset.attrs['h']

        frame_name = TEMPLATE_FRAME.format(frame_numb)

        markup.add_blob(frame_name, x, y, w, h, {'img_name': img_name})

    markup.save(out_path)


def prepare_for_markup(hdf5_path, out_name, hdf5_image_path='image'):

    out_dir = out_name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    image_dir = os.path.join(out_dir, 'imgs')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    marking_path = os.path.join(out_dir, 'marking.json')

    print("Creating markup file....")
    save_markup(hdf5_path, marking_path, hdf5_image_path)

    frame_numbers = set()
    in_db = h5py.File(hdf5_path)
    for img_name, img_dset in in_db[hdf5_image_path].items():
        frame_numbers.add(int(img_dset.attrs['frame_number']))

    print("Saving frames to image dir....")
    save_frames_to_dir(frame_numbers, image_dir)

    print("Zipping...")
    shutil.make_archive(out_name, 'zip', out_dir)
    shutil.move('{}.zip'.format(out_name), '{}.gml_hockey'.format(out_name))

    print("Removing temporary files...")
    shutil.rmtree(out_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_file', type=str, nargs='?', help='path to hdf5 file with image data')
    parser.add_argument('out_name', type=str, nargs='?', help='zip file name to save result')
    parser.add_argument('--path', type=str, default='/image', help='path to image group in hdf5 file, default="/image"')

    args = parser.parse_args()
    prepare_for_markup(args.hdf5_file, args.out_name, args.path)


if __name__ == '__main__':
    main()

