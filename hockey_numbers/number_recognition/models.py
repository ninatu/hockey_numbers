#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neural network number classification model"""

from caffe_model.model import CaffeNetworkModel
from caffe_model.data_layers import DataLayer
from caffe_model.other_layers import InnerProductLayer
from caffe_model.other_layers import DropoutLayer
from caffe_model.other_layers import AccuracyLayer
from caffe_model.other_layers import SoftmaxWithLossLayer
from caffe_model.other_layers import SoftmaxLayer
import os.path as osp

class BaseModel:

    """Abstract model class"""

    #MODEL_NAME = None ???
    WEIGHTS_PATH = None
    MODEL_PATH = None

    def get(self):
        pass


class VGG16(BaseModel):

    # MODEL_NAME = 'vgg16' ???
    MODEL_FILE = 'xxx.prototxt'
    BASEMODEL_FILE = 'VGG_ILSVRC_16_layers.prototxt'

    def _create_model(self):

        def get_pretrain_model():
            """load from prototxt VGG convolution layers"""

            model_file = osp.join('models', 'vgg16', VGG16.BASEMODEL_FILE)
            return CaffeNetworkModel(model_file)

        def add_common_inner_layers(model):
            """add inner product layer after model
            return name last layer"""

            model.add_layer(InnerProductLayer("n_fc1", num_output=1024, lr_mult=10),
                            parent_layer="pool5")
            model.add_layer(DropoutLayer("n_drop1", dropout_ratio=0.5),
                            parent_layer="n_fc1")

            return "n_fc1"

        def add_train_layers(model, last_layer):
            """add data train/test layer and softmax with loss layers"""

            model.add_layer(DataLayer('train_data', num_slots_out=2),
                            named_slots_out=[('data', 0), ('labels', 1)],
                            phase='train')
            model.add_layer(DataLayer('test_data', num_slots_out=2),
                            named_slots_out=[('data', 0), ('labels', 1)],
                            phase='test')
            """model.add_layer(AccuracyLayer("n_accuracy"),
                            parent_layer=last_layer)

            model.add_layer(SoftmaxWithLossLayer("n_loss"),
                            parent_layer=last_layer)
            """

        def add_deploy_layers(model, last_layer):
            """add data layer and softmax"""

            model.add_layer(DataLayer("deploy_data", num_slots_out=1),
                            named_slots_out=[('data', 0)])
            model.add_layer(SoftmaxLayer("n_loss"),
                            parent_layer=last_layer)

        """creation train/test net"""
        train_model = get_pretrain_model()
        last_layer = add_common_inner_layers(train_model)
        add_train_layers(train_model, last_layer)

        """creation deploy net"""
        #deploy_model = get_pretrain_model()
        #last_layer = add_common_inner_layers(deploy_model)
        #add_deploy_layers(deploy_model, last_layer)
        deploy_model = None
        return train_model.get_net_params(), \
               None #deploy_model.get_net_params()

    """
    def load_model(self):
        net = caffe.Net(MODEL_FILE, PRETRAINED__FILE, caffe.TRAIN)
        self.net = net
    """