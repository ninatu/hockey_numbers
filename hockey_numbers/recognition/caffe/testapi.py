import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
from models import VGG16
from models import ClassificationType
caffe.set_device(0)
caffe.set_mode_gpu()
model = VGG16(type=ClassificationType.BINARY)
#model.train("/home/GRAPHICS2/19n_tul/data/lmdb/number_not_number/train", "/home/GRAPHICS2/19n_tul/data/lmdb/number_not_number/test")
model.train("/tmp/train", "/tmp/train")
