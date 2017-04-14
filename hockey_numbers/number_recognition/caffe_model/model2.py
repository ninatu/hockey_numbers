from caffe.proto import caffe_pb2
from google.protobuf import text_format
from copy import deepcopy

from caffe_model.layers import CaffeProtoLayer
import datetime

class CaffeNetworkModel(object):
    def __init__(self, prototxt_path=None, name=None)
        self._layers = []
        self._layers_phases = dict()

        if name is None:
            name = "GeneratedModel_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
        self._name = name

        if prototxt_path is not None:
            self._layers = parse_model_prototxt(prototxt_path)

    def add_layer(self, layer, top_list=[], bottom_list=[]):
        if len(top_list):
            layer.params.top.extend(top_list)
        if len(bottom_list):
            layer.params.bottom

            top_layer = self.find_layer(top_layer)
            assert top_layer is None



        self._layers.append(layer

    def find_layer(self, layer_name):
        for layer in self._layers:
            if layer.name == layer_name:
                return layer
        return None


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
