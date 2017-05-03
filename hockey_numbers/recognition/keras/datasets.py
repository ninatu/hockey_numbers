import os.path as osp
import os
import shutil
import random
import scipy.misc
import numpy as np
from enum import Enum

from models import ClassificationType

DATA_FOLDER = 'data'
DIR_NOT_NUMBER = 'not_number'
#DATA_FOLDER = '/home/GRAPHICS2/19n_tul/data'

def get_dirs(path):
    files = os.listdir(path)
    return list(filter(lambda x: osp.isdir(osp.join(path, x)), files))


def get_files(path):
    files = os.listdir(path)
    return list(filter(lambda x: osp.isfile(osp.join(path, x)), files))

def move_dir2dir(indir, outdir, count=None):
    files = os.listdir(indir)
    if count:
        random.shuffle(files)
        files = files[:count]

    for file in files:
        os.rename(osp.join(indir, file), osp.join(outdir, file))


class BaseDataset:
    def __init__(self, dataset_dir):
        self._name = dataset_dir
        self._data_path = osp.join(DATA_FOLDER, dataset_dir)
        self._train_path = osp.join(self._data_path, 'train')
        self._test_path = osp.join(self._data_path, 'test')
        self._img_path = osp.join(self._data_path, 'img')

    @property
    def name(self):
        return self._name

    @property
    def is_prepared(self):
        return osp.exists(self._train_path)

    @property
    def train_directory(self):
        return self._train_path

    @property
    def test_directory(self):
        return self._test_path

    def numpy_sample(self, shape, max_count=5000):

        classes = get_dirs(self._train_path)
        count_per_class = int(max_count / len(classes))

        sample = []
        for _class in classes:
            train_dir = osp.join(self._train_path, _class)
            test_dir = osp.join(self._test_path, _class)
            files = [osp.join(train_dir, file) for file in get_files(train_dir)]
            files.extend([osp.join(test_dir, file) for file in get_files(test_dir)])
            random.shuffle(files)
            for file in files[:count_per_class]:
                img = scipy.misc.imread(file, mode='RGB' if shape[2] == 3 else 'L')
                img = scipy.misc.imresize(img, (shape[0], shape[1]))
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
            for number in range(0, 100):
                img_number_path = osp.join(self._img_path, str(number))
                train_number_path = osp.join(self._train_path, str(number))
                test_number_path = osp.join(self._test_path, str(number))

                os.mkdir(train_number_path)
                os.mkdir(test_number_path)
                self._split(train_number_path, test_number_path, img_number_path, test_per)
        elif type == ClassificationType.BINARY:
            train_not_number_dir = osp.join(DATA_FOLDER, DIR_NOT_NUMBER, 'train')
            test_not_number_dir = osp.join(DATA_FOLDER, DIR_NOT_NUMBER, 'test')

            count_class_0 = len(get_files(train_not_number_dir)) + len(get_files(test_not_number_dir))
            count_class_1 = 0
            for number in range(1, 100):
                img_number_path = osp.join(self._img_path, str(number))
                count_class_1 += len(get_files(img_number_path))

            count_for_class = min(count_class_0, count_class_1)
            print(count_for_class)

            for i in range(0, 2):
                os.mkdir(osp.join(self._train_path, str(i)))
                os.mkdir(osp.join(self._test_path, str(i)))

            # CLASS 0
            move_dir2dir(train_not_number_dir, osp.join(self._train_path, '0'),
                         count=int(count_for_class * (1 - test_per)))
            move_dir2dir(test_not_number_dir, osp.join(self._test_path, '0'),
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

        # if train is prepared for binary classification
        if len(classes) == 2:
            # MOVE FROM TRAIN
            numbers_dir = osp.join(self._train_path, '1')
            for _class in get_dirs(numbers_dir):
                move_dir2dir(osp.join(numbers_dir, _class), osp.join(self._img_path, _class))

            not_number_dir = osp.join(self._train_path, '0')
            move_dir2dir(not_number_dir, osp.join(DATA_FOLDER, DIR_NOT_NUMBER, 'train'))

            # MOVE FROM TEST
            numbers_dir = osp.join(self._test_path, '1')
            for _class in get_dirs(numbers_dir):
                move_dir2dir(osp.join(numbers_dir, _class), osp.join(self._img_path, _class))

            not_number_dir = osp.join(self._test_path, '0')
            move_dir2dir(not_number_dir, osp.join(DATA_FOLDER, DIR_NOT_NUMBER, 'test'))
        elif len(classes) == 100:
            for _class in classes:
                move_dir2dir(osp.join(self._train_path, _class), osp.join(self._img_path, _class))
                move_dir2dir(osp.join(self._test_path, _class), osp.join(self._img_path, _class))
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
            os.rename(src, dst)

        for file in files[n:]:
            src = osp.join(input_dir, file)
            dst = osp.join(train_dir, file)
            os.rename(src, dst)


class SynthNumber(BaseDataset):
    DATA_PATH = 'synth_number'

    def __init__(self):
        super(SynthNumber, self).__init__(SynthNumber.DATA_PATH)


class SynthText(BaseDataset):
    DATA_PATH = 'synth_text'

    def __init__(self):
        super(SynthText, self).__init__(SynthText.DATA_PATH)


class SynthAll(BaseDataset):
    DATA_PATH = 'synth_all'

    def __init__(self):
        super(SynthAll, self).__init__(SynthAll.DATA_PATH)


class RealDset(BaseDataset):
    DATA_PATH = 'real'

    def __init__(self):
        super(RealDset, self).__init__(RealDset.DATA_PATH)


class SynthNumberCrop(BaseDataset):
    DATA_PATH = 'synth_number_crop'

    def __init__(self):
        super(SynthNumberCrop, self).__init__(SynthNumberCrop.DATA_PATH)


class SynthTextCrop(BaseDataset):
    DATA_PATH = 'synth_text_crop'

    def __init__(self):
        super(SynthTextCrop, self).__init__(SynthTextCrop.DATA_PATH)


class SynthCrop(BaseDataset):
    DATA_PATH = 'synth_crop'

    def __init__(self):
        super(SynthCrop, self).__init__(SynthCrop.DATA_PATH)


class RealCrop(BaseDataset):
    DATA_PATH = 'real_crop'

    def __init__(self):
        super(RealCrop, self).__init__(RealCrop.DATA_PATH)


class DatasetType(Enum):
    synth_text = 'synth_text'
    synth_number = 'synth_number'
    synth = 'synth_all'
    real = 'real'
    synth_text_crop = 'synth_text_crop'
    synth_number_crop = 'synth_number_crop'
    synth_crop = 'synth_crop'
    real_crop = 'real_crop'


datasets = {DatasetType.synth_text: SynthText,
            DatasetType.synth_number: SynthNumber,
            DatasetType.synth: SynthAll,
            DatasetType.synth_text_crop: SynthTextCrop,
            DatasetType.synth_number_crop:SynthNumberCrop,
            DatasetType.synth_crop: SynthCrop,
            DatasetType.real: RealDset,
            DatasetType.real_crop: RealCrop}
