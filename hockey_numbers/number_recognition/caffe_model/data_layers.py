# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe_model.layers import PythonLayer
from caffe_model.layers import BaseLayer
from caffe.proto import caffe_pb2

DB = {"LEVELDB" : caffe_pb2.DataParameter.LEVELDB,
      "LMDB" : caffe_pb2.DataParameter.LMDB}
"""
class PythonDataLayer(PythonLayer):
    def __init__(self, name, num_classes):

        layer_params = {'num_classes': num_classes}
        super(PythonDataLayer, self).__init__(name, 'layers.roi_data_layer.RoIDataLayer',
                                              layer_params, 0, 4)

        for slot, dim in zip(self.slots_out, [[1, 3, 224, 224], [1, 5], [1,5], [1,5]]):
            slot.dim = dim

    def slots_out_names(self):
        return ['data', 'im_info', 'gt_boxes', 'ignored_boxes']
"""

class DataLayer(BaseLayer):
    def __init__(self, name, source="",
                 batch_size="", backend = "LMDB",
                 num_slots_out=2, scale=1,
                 mirror=False, crop_size=0,
                 mean_file=None, mean_value=None):

        super(DataLayer, self).__init__(name, 'Data', 0, num_slots_out)
        self._inplace = False
        self._num_slot_out = num_slots_out

        #self._params.data_param.source = source
        #self._params.data_param.batch_size = batch_size
        self._params.data_param.backend = DB[backend]

        if mean_file:
            self.params.transform_param.mean_file = mean_file

    def slots_out_names(self):
        return ['data', 'labels'][:self._num_slot_out]
