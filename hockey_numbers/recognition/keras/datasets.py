import os.path as osp
import os
import shutil
import random
import scipy.misc
import numpy as np
from enum import Enum

DATA_FOLDER = 'data'


def get_dirs(path):
    files = os.listdir(path)
    return list(filter(lambda x: osp.isdir(osp.join(path, x)), files))


def get_files(path):
    files = os.listdir(path)
    return list(filter(lambda x: osp.isfile(osp.join(path, x)), files))


class BaseDataset:
    def __init__(self, dataset_dir):
        self._name = dataset_dir
        self._data_path = osp.join(DATA_FOLDER, dataset_dir)
        self._train_path = osp.join(self._data_path, 'train')
        self._test_path = osp.join(self._data_path, 'test')

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

    def prepare(self, test=0.2):
        #if first split
        if not osp.exists(self._train_path):
            classes = get_dirs(self._data_path)
            for _class in classes:
                os.makedirs(osp.join(self._train_path, _class), exist_ok=True)
                os.makedirs(osp.join(self._test_path, _class), exist_ok=True)

            for _class in classes:
                for file in get_files(osp.join(self._data_path, _class)):
                    in_path = osp.join(self._data_path, _class, file)
                    out_path = osp.join(self._train_path, _class, file)
                    shutil.move(in_path, out_path)
                shutil.rmtree(osp.join(self._data_path, _class))

        classes = get_dirs(self._train_path)
        for _class in classes:
            self._split(_class, test)

    def _split(self, _class, test_per):
        train_dir = osp.join(self._train_path, _class)
        test_dir = osp.join(self._test_path, _class)

        files = [osp.join(train_dir, file)
                 for file in get_files(train_dir)]
        files.extend([osp.join(test_dir, file)
                      for file in get_files(test_dir)])

        random.shuffle(files)
        n = int(len(files) * test_per)
        for file in files[:n]:
            dst = osp.join(test_dir, osp.basename(file))
            if file != dst:
                os.rename(file, dst)

        for file in files[n:]:
            dst = osp.join(train_dir, osp.basename(file))
            if file != dst:
                os.rename(file, dst)

    #def get(self):
    #    return self._train_path, self._test_path


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
