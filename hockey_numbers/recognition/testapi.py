import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
from models import VGG16
from models import ClassificationType
model = VGG16(type=ClassificationType.NUMBERS)
model.train("/media/nina/Seagate Backup Plus Drive/hockey/lmdbs/hockey_color40_train_lmdb/", 
           "/media/nina/Seagate Backup Plus Drive/hockey/lmdbs/hockey_color40_test_lmdb/")
