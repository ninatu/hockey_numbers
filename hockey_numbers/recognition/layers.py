
from caffe.proto import caffe_pb2
from enum import Enum

class BaseLayer:
    def __init__(self, name, layer_type):
        self._params = caffe_pb2.LayerParameter()
        self._params.name = name
        self._params.type = layer_type

class ProtoLayer(BaseLayer):
    def __init__(self, layer_params):
        super(ProtoLayer, self).__init__("", "")
        self._params = layer_params

class Convolution(BaseLayer):
    def __init__(self, name, num_output, kernel_size,
                 bias_term=True, pad=0,
                 stride=1, weight_filler, bias_filler, lr_mult):
        super(Convolution, self).__init__(name, LayerType.CONVOLUTION)

        self._params.convolution_param.num_output = num_output
        self._params.convolution_param.kernel_size.extend([kernel_size])
        self._params.convolution_param.bias_term = bias_term
        self._params.convolution_param.pad.extend([pad])
        self._params.convolution_param.stride.extend([stride])

        self._params.convolution_param.weight_filler.MergeFrom(weight_filler.to_proto())
        self._params.convolution_param.bias_filler.MergeFrom(bias_filler.to_proto())

        weight_blob_param = caffe_pb2.ParamSpec(lr_mult=1 * lr_mult)
        bias_blob_param = caffe_pb2.ParamSpec(lr_mult=2 * lr_mult)

        self._params.param.extend([weight_blob_param, bias_blob_param]

class MaxPooling(BaseLayer):
    def __init__(self, name, kernel_size, stride):
        super(MaxPooling, self).__init__(name, "Pooling", 1, 1)
        self._params.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
        self._params.pooling_param.kernel_size = kernel_size
        self._params.pooling_param.stride = kernel_size
        self._inplace = True

class AveragePooling(BaseLayer):
    def __init__(self, name, kernel_size, stride):
        super(AveragePooling, self).__init__(name, "Pooling", 1, 1)
        self._params.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
        self._params.pooling_param.kernel_size = kernel_size
        self._params.pooling_param.stride = kernel_size
        self._inplace = True



class LayerType(Enum):
    CONVOLUTION = "Convolution"

