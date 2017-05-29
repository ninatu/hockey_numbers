#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path as osp
import os
import numpy as np
import shutil
import random
import tqdm
import scipy.misc
import hashlib
import base64


def get_subdirs(in_dir):
    subdirs = os.listdir(in_dir)
    subdirs = filter(lambda x: osp.isdir(osp.join(in_dir, x)), subdirs)
    return list(subdirs)


def get_files(in_dir):
    files = os.listdir(in_dir)
    files = filter(lambda x: osp.isfile(osp.join(in_dir, x)), files)
    return list(files)


def create_subdirs(in_dir, subdirs):
    for subdir in subdirs:
        ensure_dir(osp.join(in_dir, subdir))


def ensure_dir(in_dir):
    if not(osp.exists(in_dir)):
        os.mkdir(in_dir)


def merge(args):
    in_dirs = args.in_dirs
    out_dir = args.out_dir
    move = args.move

    subdirs = set()
    for in_dir in in_dirs:
        subdirs.update(get_subdirs(in_dir))

    ensure_dir(out_dir)
    create_subdirs(out_dir, subdirs)

    for in_dir in tqdm.tqdm(in_dirs):
        for subdir in tqdm.tqdm(get_subdirs(in_dir)):
            for file in get_files(osp.join(in_dir, subdir)):
                in_path = osp.join(in_dir, subdir, file)
                out_path = osp.join(out_dir, subdir, file)
                if move:
                    shutil.move(in_path, out_path)
                else:
                    shutil.copy(in_path, out_path)
                   
                    
def sample_dataset(args):
    in_dir = args.in_dir
    out_dir = args.out_dir
    count = args.count

    subdirs = get_subdirs(in_dir)
    count_per_subdir = int(count / len(subdirs))

    ensure_dir(out_dir)
    create_subdirs(out_dir, subdirs)

    for subdir in tqdm.tqdm(get_subdirs(in_dir)):
        files = get_files(osp.join(in_dir, subdir))
        random.shuffle(files)
        files = files[:count_per_subdir]
        for file in files:
            in_path = osp.join(in_dir, subdir, file)
            out_path = osp.join(out_dir, subdir, file)
            shutil.copy(in_path, out_path)


def split_dataset(args):
    in_dir = args.in_dir
    out_dir1, out_dir2 = args.out_dirs

    subdirs = get_subdirs(in_dir)

    ensure_dir(out_dir1)
    ensure_dir(out_dir2)
    create_subdirs(out_dir1, subdirs)
    create_subdirs(out_dir2, subdirs)

    count_files = 0
    for subdir in get_subdirs(in_dir):
        count_files += len(get_files(osp.join(in_dir, subdir)))

    random.shuffle(subdirs)
    cur_count = 0
    for subdir in tqdm.tqdm(subdirs):
        files = get_files(osp.join(in_dir, subdir))
        cur_count += len(files)
        if cur_count < count_files / 2:
            out_dir = out_dir1
        else:
            out_dir = out_dir2
        for file in files:
            in_path = osp.join(in_dir, subdir, file)
            out_path = osp.join(out_dir, subdir, file)
            shutil.copy(in_path, out_path)


def balance(args):
    in_dir = args.in_dir
    out_dir = args.out_dir

    subdirs = get_subdirs(in_dir)
    counts_files = [len(get_files(osp.join(in_dir, sub_dir)))for sub_dir in subdirs]
    median_count = int(np.median(np.array(counts_files)))

    if out_dir is not None:
        ensure_dir(out_dir)
        create_subdirs(out_dir, subdirs)

    for subdir in tqdm.tqdm(subdirs):
        all_files = get_files(osp.join(in_dir, subdir))
        good_files = all_files
        bad_files = []

        if len(good_files) > median_count:
            random.shuffle(all_files)
            good_files = all_files[:median_count]
            bad_files = all_files[median_count:]


        if out_dir is not None:
            for file in good_files:
                in_path = osp.join(in_dir, subdir, file)
                out_path = osp.join(out_dir, subdir, file)
                shutil.copy(in_path, out_path)
        else:
            for file in bad_files:
                path = osp.join(in_dir, subdir, file)
                os.remove(path)


def crop_square(img):
    img = scipy.misc.imresize(img, (128, 64))
    #h = img.shape[0]
    #w = img.shape[1]
    #return img[:int(h/2)]#
    return img[14:58, 10:54]


def crop(args):
    in_dir = args.in_dir
    out_dir = args.out_dir

    ensure_dir(out_dir)
    for (dirpath, dirnames, files) in tqdm.tqdm(os.walk(in_dir)):
        subdirs = os.path.relpath(dirpath, in_dir)
        os.makedirs(osp.join(out_dir, subdirs), exist_ok=True)

        for file in files:
            in_path = osp.join(in_dir, subdirs, file)
            out_path = osp.join(out_dir, subdirs, file)

            img = scipy.misc.imread(in_path)
            img = crop_square(img)
            scipy.misc.imsave(out_path, img)


def svhn_crop(args):
    def crop_with_pad(img, x1, y1, x2, y2):
        h, w, c = img.shape
        max_pad = int(max(abs(min(x1, 0)), abs(min(y1, 0)), abs(min(w - x2, 0)), abs(min(h - y2, 0))))
        if max_pad > 200:
            return None
        img = np.pad(img, pad_width=max_pad, mode='edge')[:, :, max_pad:max_pad + 3]
        return img[y1 + max_pad:y2 + max_pad, x1 + max_pad:x2 + max_pad]

    def sample_crop(img, x, y, w, h, pad_min, pad_max):
        pad_w1 = int(w * (pad_min + random.random() + (pad_max - pad_min)))
        pad_h1 = int(h * (pad_min + random.random() + (pad_max - pad_min)))
        pad_w2 = int(w * (pad_min + random.random() + (pad_max - pad_min)))
        pad_h2 = int(h * (pad_min + random.random() + (pad_max - pad_min)))

        x1 = x - pad_w1
        x2 = x + w + pad_w2
        y1 = y - pad_h1
        y2 = y + h + pad_h2
        return crop_with_pad(img, x1, y1, x2, y2)

    def md5_hash(arr):
        arr = np.array(arr, dtype=arr.dtype)
        md5 = hashlib.md5()
        arr = base64.b64encode(arr)
        md5.update(arr)
        return md5.hexdigest()

    out_dir = args.out_dir
    ensure_dir(out_dir)
    for i in range(1, 101):
        ensure_dir(osp.join(out_dir, str(i)))

    with open(args.gt) as gt_file:
        for line in tqdm.tqdm(gt_file.readlines()):
            parts = line.strip().split()
            img_path = parts[0]
            img_path = osp.join(args.img_dir, img_path)
            number, y, x, y2, x2 = list(map(int, parts[1:]))

            img = scipy.misc.imread(img_path)
            if len(img.shape) != 3 or img.shape[2] != 3:
                continue
            img = sample_crop(img, x, y, x2 - x, y2 - y, args.min, args.max)
            if img is None:
                continue
            img_name = '{}.png'.format(md5_hash(img))
            img_name = osp.join(out_dir, str(number), img_name)
            if img.shape[0] * img.shape[1] < 400:
                continue
            scipy.misc.imsave(img_name, img)



def main():
    parser = argparse.ArgumentParser(description='Dataset tools')
    subparsers = parser.add_subparsers()

    parser_merge = subparsers.add_parser('merge', help='merge datasets')
    parser_merge.add_argument('in_dirs', type=str, nargs='*', help='input dirs')
    parser_merge.add_argument('-o', '--out_dir', type=str, required=True, help='output dir')
    parser_merge.add_argument('--move', action='store_true', default=False, help='move images')
    parser_merge.set_defaults(func=merge)

    parser_balance = subparsers.add_parser('balance', help='balance distribution images in subdirs')
    parser_balance.add_argument('in_dir', type=str, nargs='?', help='input dir')
    parser_balance.add_argument('-o', '--out_dir', type=str, nargs='?', required=False,  help='output dir if need')
    parser_balance.set_defaults(func=balance)

    parser_balance = subparsers.add_parser('crop',
                                           help='making the image square by cutting top half for all image in dir')
    parser_balance.add_argument('in_dir', type=str, nargs='?', help='input dir')
    parser_balance.add_argument('out_dir', type=str, nargs='?', help='output dir')
    parser_balance.set_defaults(func=crop)
    
    parser_sample = subparsers.add_parser('sample', help='sample dataset')
    parser_sample.add_argument('in_dir', type=str, nargs='?', help='input dir')
    parser_sample.add_argument('-o', '--out_dir', type=str, required=True, help='output dir')
    parser_sample.add_argument('-c', '--count', type=int, required=True, help='count sampled images')
    parser_sample.set_defaults(func=sample_dataset)

    parser_split = subparsers.add_parser('split', help='split dataset on two by numbers')
    parser_split.add_argument('in_dir', type=str, nargs='?', help='input dir')
    parser_split.add_argument('-o', '--out_dirs', type=str, nargs=2, required=True, help='two output dirs')
    parser_split.set_defaults(func=split_dataset)

    parser_svhn = subparsers.add_parser('svhn', help='crop images from svhn')
    parser_svhn.add_argument('img_dir', type=str, nargs='?', help='image dir')
    parser_svhn.add_argument('gt', type=str, nargs='?', help='path to gt')
    parser_svhn.add_argument('--min', type=float, nargs='?', help='min persent pad')
    parser_svhn.add_argument('--max', type=float, nargs='?', help='max persent pad')
    parser_svhn.add_argument('-o', '--out_dir', type=str, nargs='?', required=True, help='output dir')
    parser_svhn.set_defaults(func=svhn_crop)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
