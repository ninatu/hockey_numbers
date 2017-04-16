import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
from models import VGG16
from models import ClassificationType
caffe.set_device(1)
caffe.set_mode_gpu()
model = VGG16(type=ClassificationType.NUMBERS)
model.train("/home/GRAPHICS2/19n_tul/data/lmdb/classes/train_big", "/home/GRAPHICS2/19n_tul/data/lmdb/classes/train_big")
