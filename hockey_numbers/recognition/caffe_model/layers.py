# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from abc import abstractmethod
from abc import ABCMeta

from caffe_model.fillers import GaussianFiller
from caffe_model.fillers import ConstantFiller
from caffe.proto import caffe_pb2
import json


class CaffeAbstractLayer(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def top(self):
        pass

    @property
    @abstractmethod
    def bottom(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    @abstractmethod
    def dynamic_params(self):
        pass

    @abstractmethod
    def set_dynamic_param(self, param, value):
        pass


class BaseLayer(CaffeAbstractLayer):
    def __init__(self, name, layer_type,
                 bottom, top, phase=None):
        self._params = caffe_pb2.LayerParameter()
        self._params.name = name
        self._params.type = layer_type
        self._params.top.extend(top)
        self._params.bottom.extend(bottom)
        if phase:
            self._params.i

    @property
    def params(self):
        return self._params

    @property
    def top(self):
        return self._params.top

    @property
    def bottom(self):
        return self._params.bottom

    @property
    def name(self):
        return self._params.name

    @property
    def dynamic_params(self):
        return self._dynamic_params

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()


class CaffeProtoLayer(BaseLayer):
    def __init__(self, params):
        super(CaffeProtoLayer, self).__init__(name="",
                                              layer_type="",
                                              bottom=params.bottom,
                                              top=params.top)
        self._params = params



DB = {"LEVELDB" : caffe_pb2.DataParameter.LEVELDB,
      "LMDB" : caffe_pb2.DataParameter.LMDB}

class DataLayer(BaseLayer):
    def __init__(self, name, top, source="",
                 batch_size="", backend = "LMDB",
                 num_slots_out=2, scale=1,
                 mirror=False, crop_size=0,
                 mean_file=None, mean_value=None):

        super(DataLayer, self).__init__(name, 'Data', [], top)
        #self._params.data_param.source = source
        #self._params.data_param.batch_size = batch_size
        self._params.data_param.backend = DB[backend]

        if mean_file:
            self.params.transform_param.mean_file = mean_file


class ConvolutionLayer(BaseLayer):
    def __init__(self, name, bottom, top,
                 num_filters, kernel_size,
                 stride=1, pad=0,
                 weight_filler=GaussianFiller(std=0.01),
                 lr_mult=1):
        super(ConvolutionLayer, self).__init__(name, 'Convolution', bottom, top)
        self._inplace = False

        self._params.convolution_param.num_output = num_filters
        self._params.convolution_param.kernel_size.extend([kernel_size])
        self._params.convolution_param.pad.extend([pad])
        self._params.convolution_param.stride.extend([stride])

        self._params.convolution_param.weight_filler.MergeFrom(weight_filler.to_proto())
        self._params.convolution_param.bias_filler.MergeFrom(ConstantFiller().to_proto())

        weight_blob_param = caffe_pb2.ParamSpec(lr_mult=1 * lr_mult)
        bias_blob_param = caffe_pb2.ParamSpec(lr_mult=2 * lr_mult)

        self._params.param.extend([weight_blob_param, bias_blob_param])

class MaxPooling(BaseLayer):
    def __init__(self, name, bottom, kernel_size, stride):
        super(MaxPooling, self).__init__(name, "Pooling", bottom, bottom)
        self._params.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
        self._params.pooling_param.kernel_size = kernel_size
        self._params.pooling_param.stride = kernel_size


class AveragePooling(BaseLayer):
    def __init__(self, name, bottom, kernel_size, stride):
        super(AveragePooling, self).__init__(name, "Pooling", bottom, bottom)
        self._params.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
        self._params.pooling_param.kernel_size = kernel_size
        self._params.pooling_param.stride = kernel_size


class InnerProductLayer(BaseLayer):
    def __init__(self, name, bottom, top,
                 num_output, bias_term=True,
                 weight_filler=GaussianFiller(std=0.01),
                 lr_mult=1):
        super(InnerProductLayer, self).__init__(name, "InnerProduct", bottom, top)

        self._params.inner_product_param.num_output = num_output
        self._params.inner_product_param.bias_term = bias_term
        self._params.inner_product_param.weight_filler.MergeFrom(weight_filler.to_proto())
        self._params.inner_product_param.bias_filler.MergeFrom(ConstantFiller().to_proto())

        weight_blob_param = caffe_pb2.ParamSpec(lr_mult=1 * lr_mult)
        bias_blob_param = caffe_pb2.ParamSpec(lr_mult=2 * lr_mult)

        self._params.param.extend([weight_blob_param, bias_blob_param])


class AccuracyLayer(BaseLayer):
    def __init__(self, name, bottom, top):
        super(AccuracyLayer, self).__init__(name, "Accuracy", bottom, top)


class DropoutLayer(BaseLayer):
    def __init__(self, name, bottom, dropout_ratio):
        super(DropoutLayer, self).__init__(name, "Dropout", bottom, bottom)

        self._params.dropout_param.dropout_ratio = dropout_ratio


class ReshapeLayer(BaseLayer):
    def __init__(self, name, bottom, reshape_dim):
        super(ReshapeLayer, self).__init__(name, "Reshape", bottom, bottom)

        self._params.reshape_param.shape.dim.extend(reshape_dim)



class SoftmaxLayer(BaseLayer):
    def __init__(self, name, bottom, top):
        super(SoftmaxLayer, self).__init__(name, "Softmax", bottom, top)


class SoftmaxWithLossLayer(BaseLayer):
    def __init__(self, name, bottom, top, loss_weight=1, ignore_label=-1):
        super(SoftmaxWithLossLayer, self).__init__(name, "SoftmaxWithLoss", bottom, top)

        self._params.loss_param.normalization = caffe_pb2.LossParameter.VALID
        self._params.loss_param.ignore_label = ignore_label
        self._params.loss_weight.extend([loss_weight])


class SmoothL1LossLayer(BaseLayer):
    def __init__(self, name, bottom, top, sigma=3, loss_weight=1):
        super(SmoothL1LossLayer, self).__init__(name, "SmoothL1Loss", bottom, top)

        self._params.loss_weight.extend([loss_weight])
        self._params.smooth_l1_loss_param.sigma = sigma



"""
class SmoothL1LossPyLayer(PythonLayer):
        def __init__(self, name, sigma=3, loss_weight=1):
            self._layer_params = {'sigma': sigma}

            super(SmoothL1LossPyLayer, self).__init__(name, 'layers.smooth_l1_loss.SmoothL1LossLayer',
                                                      self._layer_params, 4, 1)
            self._inplace = False
            self._params.loss_weight.extend([loss_weight])

        def slots_out_names(self):
            return ['']





class PythonLayer(BaseLayer):
    def __init__(self, name, python_class_name, layer_params, num_slots_in, num_slots_out):
        super(PythonLayer, self).__init__(name, 'Python',
                                          num_slots_in, num_slots_out)

        splitted = python_class_name.split('.')

        self._params.python_param.module = '.'.join(splitted[:-1])
        self._params.python_param.layer = splitted[-1]
        self.update_layer_params(layer_params)

    def update_layer_params(self, layer_params):
        self._params.python_param.param_str = json.dumps(layer_params)
"""