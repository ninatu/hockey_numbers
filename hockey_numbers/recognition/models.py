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

class ClassificationType(Enum):
    NUMBERS = 'numbers'
    BINARY = 'binary'

class BaseModel(object):

    def __init__(self, type):
        self._type = type
        if type == ClassificationType.NUMBERS:
            self._num_out = 100
        elif type == ClassificationType.BINARY:
            self._num_out = 2
        else:
            raise NotImplementedError(type)

    """Abstract model class"""

    @property
    def num_out(self):
        return self._num_out

    def get(self):
        pass


class VGG16(BaseModel):

    MODEL_DIR = 'vgg16'
    MODEL_FILE = 'numbers_{:s}_{:s}.prototxt'
    BASEMODEL_WEIGHTS = 'VGG_ILSVRC_16_layers.caffemodel'
    BASEMODEL_FILE = 'VGG_ILSVRC_16_layers.prototxt'


    def __init__(self, type):
        super(VGG16, self).__init__(type)

        self._train_proto_net, self._deploy_proto_net = self.load_model()

    def _create_model(self):

        def get_pretrain_model():
            """load from prototxt VGG convolution layers"""

            model_file = osp.join('models', VGG16.MODEL_DIR, VGG16.BASEMODEL_FILE)
            return CaffeNetworkModel(model_file)

        def add_common_inner_layers(model):
            """add inner product layer after model
            return name last layer"""

            model.add_layer(InnerProductLayer('n_fc1', bottom=['pool5'], top=['n_fc1'],
                                              num_output=1024, lr_mult=10))
            model.add_layer(DropoutLayer('n_drop1', bottom=['n_fc1'],
                                         dropout_ratio=0.5))

            model.add_layer(InnerProductLayer('n_fc2', bottom=['n_fc1'], top=['n_fc2'],
                                              num_output=self.num_out, lr_mult=10))

            return 'n_fc2'

        """creation train/test net"""

        train_model = CaffeNetworkModel()
        train_model.add_layer(DataLayer('data', top=['data', 'label'], phase='train', batch_size=64))
        train_model.add_layer(DataLayer('data', top=['data', 'label'], phase='test', batch_size=64))

        train_model.merge(get_pretrain_model())

        last_layer = add_common_inner_layers(train_model)

        train_model.add_layer(AccuracyLayer("n_accuracy", bottom=[last_layer, 'label'],
                                      top=['accuracy'], phase='test'))
        train_model.add_layer(SoftmaxWithLossLayer("n_loss", bottom=[last_layer, 'label'],
                                             top=['loss']))

        """creation deploy net"""
        deploy_model = CaffeNetworkModel()
        deploy_model.add_layer(DataLayer("deploy_data", top=['data'], batch_size=64))

        deploy_model.merge(get_pretrain_model())

        last_layer = add_common_inner_layers(deploy_model)

        deploy_model.add_layer(SoftmaxLayer("n_loss", bottom=[last_layer], top=['loss']))

        return train_model.get_net_params(), \
               deploy_model.get_net_params()


    def load_model(self):

        model_train_file = VGG16.MODEL_FILE.format(str(self._type), 'train_test')
        model_train_file = osp.join('models', VGG16.MODEL_DIR, model_train_file)

        model_deploy_file = VGG16.MODEL_FILE.format(str(self._type), 'deploy')
        model_deploy_file = osp.join('models', VGG16.MODEL_DIR, model_deploy_file)


        if True: #not osp.exists(model_train_file):
            train_proto_net, deploy_proto_net = self._create_model()

            with open(model_train_file, 'w') as tr_f:
                tr_f.write(str(train_proto_net))

            with open(model_deploy_file, 'w') as dep_f:
                dep_f.write(str(deploy_proto_net))
        else:
            train_proto_net = CaffeNetworkModel(model_train_file).get_net_params()
            deploy_proto_net = CaffeNetworkModel(model_deploy_file).get_net_params()

        return train_proto_net, deploy_proto_net
        #self._train_model = caffe.Net(model_train_file, caffe.TRAIN)
        #self._test_model = caffe.Net(model_train_file, caffe.TEST)
        #self._deploy_model = caffe.Net(model_deploy_file, caffe.TEST)

    def get_train_proto_net(self, train_dset, valid_dset):
        net_param = deepcopy(self._train_proto_net)

        layers = net_param.layer
        layer_train_data = None
        layer_test_data = None
        for layer in layers:
            if layer.name == 'data':
                if layer.include[0].phase == caffe_pb2.TRAIN:
                    layer_train_data = layer
                if layer.include[0].phase == caffe_pb2.TEST:
                    layer_test_data = layer
        assert layer_train_data is not None
        assert layer_test_data is not None
    
        if layer_train_data.type == "Data":
            layer_train_data.data_param.source = train_dset
            layer_test_data.data_param.source = valid_dset
        else:
            layer_train_data.image_data_param.source = train_dset
            layer_test_data.image_data_param.source = valid_dset

        #net_param.input_dim.extend([64, 3, 40, 40])
        return net_param

    """
    def get_test_proto_net(self, train_dset, test_dset):
        net_param = deepcopy(self._train_proto_net)

        layers = net_param.layer
        layer_train_data = None
        layer_test_data = None
        for layer in layers:
            if layer.name == 'train_data':
                layer_train_data = layer
            if layer.name == 'test_data':
                layer_test_data = layer

        assert layer_train_data is None
        assert layer_test_data is None

        layer_train_data.source = train_data
        layer_test_data.source = valid_data
        return net_param
    """

    def train(self, train_dset, test_dset, pretrained_path=None):
        snapshot_path = osp.join('models', VGG16.MODEL_DIR, str(self._type))
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

        solverproto = SolverProto(net_params=self.get_train_proto_net(train_dset, test_dset),
                                   params=solver_params)
        solver = Solver(solverproto)
        if pretrained_path is None:
            pretrained_path = osp.join('models', VGG16.MODEL_DIR, VGG16.BASEMODEL_WEIGHTS)
        solver.solve(pretrained_path)
        solverproto.close()

        solver.get_net().save(osp.join(snapshot_path, 'weights'))

    def deploy(self, dataset_path):
        pass

    def test(self, dataset_path):
        pass
