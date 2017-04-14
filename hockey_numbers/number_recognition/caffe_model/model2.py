from caffe.proto import caffe_pb2
from google.protobuf import text_format
from copy import deepcopy

from caffe_model.layers import CaffeProtoLayer
import datetime

class CaffeNetworkModel(object):
    def __init__(self, prototxt_path=None, name=None):
        self._layers = []
        self._layers_phases = dict()

        if name is None:
            name = "GeneratedModel_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
        self._name = name

        if prototxt_path is not None:
            self._layers = parse_model_prototxt(prototxt_path)

    def add_layer(self, layer):
        self._layers.append(layer)

    def merge(self, model):
        for layer in model._layers:
            self._layers.append(layer)

    def get_net_params(self):
        params = caffe_pb2.NetParameter()

        proto_layers = []
        for layer in self._layers:
            proto_layer = deepcopy(layer.params)
            proto_layers.append(proto_layer)

        params.layer.extend(proto_layers)
        params.name = self._name

        return params


def parse_model_prototxt(prototxt_path):
    net_params = load_prototxt(prototxt_path)

    layers = []

    for layer_params in net_params.layer:
        layer = CaffeProtoLayer(layer_params)
        layers.append(layer)

    return layers


def load_prototxt(prototxt_path):
    params = caffe_pb2.NetParameter()

    with open(prototxt_path, "r") as f:
        text_format.Merge(str(f.read()), params)

    return params
