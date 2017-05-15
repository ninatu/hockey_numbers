#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import ClassificationType
from constants import DATA_FOLDER, DIR_NOT_NUMBER, DIR_NOT_NUMBER_CROP, DIR_NOT_NUMBER_HALF

import os.path as osp
import os
import shutil
import random
import scipy.misc
import numpy as np
from enum import Enum


def get_dirs(path):
    files = os.listdir(path)
    return list(filter(lambda x: osp.isdir(osp.join(path, x)), files))


def get_files(path):
    files = os.listdir(path)
    return list(filter(lambda x: osp.isfile(osp.join(path, x)), files))


def recursively_get_files(path):
    files = []
    for root_dir, subdirs, subfiles in os.walk(path):
        relpath = osp.relpath(root_dir, path)
        for subfile in subfiles:
            files.append(osp.join(relpath, subfile))
    return files


def unlink_files_in_dir(indir):
    files = os.listdir(indir)
    for file in files:
        os.unlink(osp.join(indir, file))


def link_files_to_dir(indir, outdir, count=None):
    files = os.listdir(indir)
    if count:
        random.shuffle(files)
        files = files[:count]
    for file in files:
        os.link(osp.join(indir, file), osp.join(outdir, file))


class BaseDataset:
    def __init__(self, dataset_dir, type=None):
        self._name = dataset_dir
        self._data_path = osp.join(DATA_FOLDER, dataset_dir)
        self._train_path = osp.join(self._data_path, 'train')
        self._test_path = osp.join(self._data_path, 'test')
        self._img_path = osp.join(self._data_path, 'img')
        if type == 'crop':
            self._not_player_path = osp.join(DATA_FOLDER, DIR_NOT_NUMBER_CROP)
        elif type == 'half':
            self._not_player_path = osp.join(DATA_FOLDER, DIR_NOT_NUMBER_HALF)
        else:
            self._not_player_path = osp.join(DATA_FOLDER, DIR_NOT_NUMBER)

    @property
    def name(self):
        return self._name

    @property
    def is_prepared(self):
        print(self._train_path)
        return osp.exists(self._train_path)

    def get_train(self, shape):
        return (self._train_path, self._get_sample(self._train_path, shape))

    def get_test(self, shape):
        return (self._test_path, self._get_sample(self._test_path, shape))

    def _get_sample(self, path, shape, max_count=20000):
        files = recursively_get_files(path)
        random.shuffle(files)
        files = files[:max_count]

        sample = []
        for file in files:
            file = osp.join(path, file)
            img = scipy.misc.imread(file, mode='RGB' if shape[2] == 3 else 'L')
            img = scipy.misc.imresize(img, (shape[0], shape[1]))
            if shape[2] == 1:
                img = img[:, :, np.newaxis]
            sample.append(img)
        return np.array(sample)

    def prepare(self, type, test_per=0.2):
        #if first split
        if not osp.exists(self._train_path):
            classes = get_dirs(self._data_path)
            os.mkdir(self._train_path)
            os.mkdir(self._test_path)
            os.mkdir(self._img_path)
            for _class in classes:
                shutil.move(osp.join(self._data_path, _class), osp.join(self._img_path, _class))

        self._clean_train_test()

        if type == ClassificationType.NUMBERS:
            for number in get_dirs(self._img_path):
                img_number_path = osp.join(self._img_path, str(number))
                train_number_path = osp.join(self._train_path, str(number))
                test_number_path = osp.join(self._test_path, str(number))

                os.mkdir(train_number_path)
                os.mkdir(test_number_path)
                self._split(train_number_path, test_number_path, img_number_path, test_per)
        elif type == ClassificationType.BINARY:
            train_not_number_dir = osp.join(self._not_player_path, 'train')
            test_not_number_dir = osp.join(self._not_player_path, 'test')

            count_class_0 = len(get_files(train_not_number_dir)) + len(get_files(test_not_number_dir))
            count_class_1 = 0
            for number in range(1, 100):
                img_number_path = osp.join(self._img_path, str(number))
                count_class_1 += len(get_files(img_number_path))

            count_for_class = min(count_class_0, count_class_1)
            print(count_for_class)

            for i in [1, 2]:
                os.mkdir(osp.join(self._train_path, str(i)))
                os.mkdir(osp.join(self._test_path, str(i)))

            # CLASS 0
            link_files_to_dir(train_not_number_dir, osp.join(self._train_path, '2'),
                         count=int(count_for_class * (1 - test_per)))
            link_files_to_dir(test_not_number_dir, osp.join(self._test_path, '2'),
                         count=int(count_for_class * test_per))

            # CLASS 1
            numbers = list(range(1, 100))
            random.shuffle(numbers)
            count = 0
            for number in numbers:
                img_number_path = osp.join(self._img_path, str(number))
                train_number_path = osp.join(self._train_path, '1', str(number))
                test_number_path = osp.join(self._test_path, '1', str(number))

                os.mkdir(train_number_path)
                os.mkdir(test_number_path)
                count += len(get_files(img_number_path))
                if count >= count_for_class:
                    break
                self._split(train_number_path, test_number_path, img_number_path, test_per)

    def _clean_train_test(self):
        classes = get_dirs(self._train_path)
        print(self._train_path)
        print(self._test_path)
        print(self._img_path)
        print(self._not_player_path)
        # if train is prepared for binary classification
        if len(classes) == 2:
            unlink_files_in_dir(osp.join(self._train_path, '2'))
            unlink_files_in_dir(osp.join(self._test_path, '2'))

            classes = get_dirs(osp.join(self._train_path, '1'))
            for _class in classes:
                unlink_files_in_dir(osp.join(self._train_path, '1', _class))
                unlink_files_in_dir(osp.join(self._test_path, '1', _class))
        elif len(classes) == 100:
            for _class in classes:
                unlink_files_in_dir(osp.join(self._train_path, _class))
                unlink_files_in_dir(osp.join(self._test_path, _class))
        elif len(classes) == 0:
            pass
        else:
            raise NotImplementedError

        for _class in get_dirs(self._train_path):
            shutil.rmtree(osp.join(self._train_path, _class))
        for _class in get_dirs(self._test_path):
            shutil.rmtree(osp.join(self._test_path, _class))

    def _split(self, train_dir, test_dir, input_dir, test_per):
        files = get_files(input_dir)
        random.shuffle(files)

        n = int(len(files) * test_per)
        for file in files[:n]:
            src = osp.join(input_dir, file)
            dst = osp.join(test_dir, file)
            os.link(src, dst)

        for file in files[n:]:
            src = osp.join(input_dir, file)
            dst = osp.join(train_dir, file)
            os.link(src, dst)


class DatasetType(Enum):
    fvsd_crop = 'fvsd_crop'
    fvsd_half = 'fvsd_half'

    risd_crop = 'risd_crop'
    risd_half = 'risd_half'
    
    naive_imp_crop = 'naive_imp_crop'
    naive_imp_half = 'naive_imp_half'

    naive_crop = 'naive_crop'
    naive_half = 'naive_half'

    full_crop = 'full_crop'
    full_half = 'full_half'

    real_crop = 'real_crop'
    real_half = 'real_half'
 
    svhn_crop = 'svhn_crop'
    svhn_half = 'svhn_half'

    realtrain_crop = 'realtrain_crop'
    realtrain_half = 'realtrain_half'
 
    realhalf_crop = 'realhalf_crop'
    realhalf_half = 'realhalf_half'
 

datasets = {DatasetType.fvsd_crop: lambda: BaseDataset('FVSD_crop', 'crop'),
            DatasetType.fvsd_half: lambda: BaseDataset('FVSD_half', 'half'),

            DatasetType.risd_crop: lambda: BaseDataset('RISD_crop', 'crop'),
            DatasetType.risd_half: lambda: BaseDataset('RISD_half', 'half'),

            DatasetType.naive_crop: lambda: BaseDataset('NaiveSD_crop', 'crop'),
            DatasetType.naive_half: lambda: BaseDataset('NaiveSD_half', 'half'),

            DatasetType.naive_imp_crop: lambda: BaseDataset('NaiveImpSD_crop', 'crop'),
            DatasetType.naive_imp_half: lambda: BaseDataset('NaiveImpSD_half', 'half'),

            DatasetType.real_crop: lambda: BaseDataset('Real_crop', 'crop'),
            DatasetType.real_half: lambda: BaseDataset('Real_half', 'half'),

            DatasetType.svhn_crop: lambda: BaseDataset('SVHN_crop', 'crop'),
            DatasetType.svhn_half: lambda: BaseDataset('SVHN_half', 'half'),

            DatasetType.full_crop: lambda: BaseDataset('FullSD_crop', 'crop'),
            DatasetType.full_half: lambda: BaseDataset('FullSD_half', 'half'),

            DatasetType.realtrain_crop: lambda: BaseDataset('RealTrain_crop', 'crop'),
            DatasetType.realtrain_half: lambda: BaseDataset('RealTrain_half', 'half'),

            DatasetType.realhalf_crop: lambda: BaseDataset('RealHalf_crop', 'crop'),
            DatasetType.realhalf_half: lambda: BaseDataset('RealTrain_half', 'half')}
