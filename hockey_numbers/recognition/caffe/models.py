#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neural network number classification model"""

from caffe_model.model import CaffeNetworkModel
from caffe_model.layers import ImageDataLayer
from caffe_model.layers import DataLayer
from caffe_model.layers import InnerProductLayer
from caffe_model.layers import DropoutLayer
from caffe_model.layers import AccuracyLayer
from caffe_model.layers import SoftmaxWithLossLayer
from caffe_model.layers import SoftmaxLayer
from caffe_model.solver import SolverProto
from caffe_model.solver import Solver
import caffe
from caffe.proto import caffe_pb2
import os
import os.path as osp
from copy import deepcopy
from enum import Enum
from abc import abstractmethod
from abc import ABCMeta

class ClassificationType(Enum):
    NUMBERS = 'numbers'
    BINARY = 'binary'

class CaffeAbstractModel:
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def type(self):
        pass

    @property
    @abstractmethod
    def train(self, train_dset, test_dset, pretrained_path=None):
        pass

    @property
    @abstractmethod
    def model_dir(self):
        pass


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, type, base_model_file,
                 base_model_weights, dir_to_save_model):

        assert isinstance(type, ClassificationType)
        self._type = type
        self._base_model_weights = base_model_weights
        self._base_model_file = base_model_file
        self._model_dir = dir_to_save_model

    @abstractmethod
    def add_common_inner_layers(self):
        pass

    @property
    def type(self):
        return self._type

    @property
    def num_out(self):
        return 100 if self.type == ClassificationType.NUMBERS else 2

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def base_model_file(self):
        return self._base_model_file
    @property
    def base_model_weights(self):
        return self._base_model_weights


    def get_base_model(self):
        """ load base model net"""

        if self._base_model_file:
            return CaffeNetworkModel(self._base_model_file)

    def get_train_proto_net(self, train_dset, valid_dset, batch_size=64):
        """creation train/test net"""

        train_model = CaffeNetworkModel()
        train_model.add_layer(DataLayer('data', top=['data', 'label'],
                                        phase='train', batch_size=batch_size,
                                        source=train_dset))
        train_model.add_layer(DataLayer('data', top=['data', 'label'],
                                        phase='test', batch_size=batch_size,
                                        source=valid_dset))

        base_model = self.get_base_model()
        if base_model:
            train_model.merge(base_model)

        last_layer = self.add_common_inner_layers(train_model)

        train_model.add_layer(AccuracyLayer("n_accuracy", bottom=[last_layer, 'label'],
                                            top=['accuracy'], phase='test'))
        train_model.add_layer(SoftmaxWithLossLayer("n_loss", bottom=[last_layer, 'label'],
                                                   top=['loss']))

        train_model.add_input_dim([batch_size, 3, 64, 32])

        self.save_net(train_model, 'train')
        return train_model.get_net_params()

    def get_deploy_proto_net(self, deploy_dset):
        """creation deploy net"""

        deploy_model = CaffeNetworkModel()
        deploy_model.add_layer(DataLayer("deploy_data", top=['data'],
                                         source=deploy_dset, batch_size=64))

        base_model = self.get_base_model()
        if base_model:
            deploy_model.merge(base_model)

        last_layer = self.add_common_inner_layers(deploy_model)

        deploy_model.add_layer(SoftmaxLayer("n_loss", bottom=[last_layer], top=['loss']))

        deploy_model.add_input_dim([batch_size, 3, 64, 32])

        self.save_net(deploy_model, 'deploy')
        return deploy_model.get_net_params()

    def get_test_proto_net(self, train_dset, valid_dset):
        raise NotImplementedError()

    def save_net(self, model, model_type):
        path_model = "{:s}_{:s}_{:s}.prototxt".format(model.name, self.type.value, model_type)
        path_model = osp.join(self.model_dir, path_model)

        with open(path_model, 'w') as fout:
            fout.write(str(model.get_net_params()))



class VGG16(BaseModel):

    def __init__(self, type):
        super(VGG16, self).__init__(type, base_model_file='models/vgg16/VGG_ILSVRC_16_layers.prototxt',
                                    base_model_weights='models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
                                    dir_to_save_model='models/vgg16/my_models')

    @abstractmethod
    def add_common_inner_layers(self, model):
        """add inner product layer after model
        return name last layer"""

        model.add_layer(InnerProductLayer('n_fc1', bottom=['pool5'], top=['n_fc1'],
                                          num_output=1024, lr_mult=10))
        model.add_layer(DropoutLayer('n_drop1', bottom=['n_fc1'],
                                     dropout_ratio=0.5))

        model.add_layer(InnerProductLayer('n_fc2', bottom=['n_fc1'], top=['n_fc2'],
                                          num_output=self.num_out, lr_mult=10))

        return 'n_fc2'

    @abstractmethod
    def train(self, train_dset, test_dset, pretrained_path=None):

        train_net = self.get_train_proto_net(train_dset, test_dset)

        snapshot_path = osp.join(self.model_dir, self.type.value)
        if not osp.exists(snapshot_path):
            os.mkdir(snapshot_path)
        snapshot_path = osp.join(snapshot_path, 'snapshot')
        if not osp.exists(snapshot_path):
            os.mkdir(snapshot_path)

        solver_params = {
            'base_lr': 0.0001,
            'lr_policy': 'step',
            'gamma': 0.1,
            'stepsize': 5000,
            'display': 50,
            'max_iter': 120000,
            'test_iter': 3,
            'test_interval': 200,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'snapshot': 10000,
            'snapshot_prefix': snapshot_path + '/' + 'sn',
            'solver_mode': 'GPU',
        }

        solver = Solver(train_net, solver_params)
        if pretrained_path is None:
            pretrained_path = self.base_model_weights

        solver.save_params(self.model_dir)
        solver.solve(pretrained_path)
        solver.save_weights(osp.join(snapshot_path, 'weights'))

    def deploy(self, dataset_path):
        pass

    def test(self, dataset_path):
        pass


