from abc import abstractmethod
from abc import ABCMeta
from enum import Enum
import datetime
import os.path as osp
import numpy as np
import sys
import json

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from model_utils import get_data_generator, compile_model
from api import evaluate_model


MODEL_DATA_DIR = '/home/GRAPHICS2/19n_tul/models'
#MODEL_DATA_DIR = 'models'


class ClassificationType(Enum):
    NUMBERS = 'numbers'
    BINARY = 'binary'


class AbstractModel(metaclass=ABCMeta):

    @abstractmethod
    def train(self, train_dset, epochs, freeze_base, test_dset):
        pass

    @abstractmethod
    def set_params(self):
        pass

    @abstractmethod
    def save(self):
        pass


class BaseModel(AbstractModel):
    def __init__(self, name, type):
        self._type = type
        self._base_name = '{}_{}'.format(name, type.value)
        self._name = '{}_{}'.format(self._base_name, datetime.datetime.now().strftime("%d_%m_%Y_%H_%M"))
        self._path = osp.join(MODEL_DATA_DIR, self.name + '_model.hdf5')

        self._model = None
        self._base_model = None

        self._input_size = None
        self._gray = False
        self._pretrained = None
        self._lr = None
        self._batch_size = None
        self._epoch_images = None

    @abstractmethod
    def _prepare_model(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def path(self):
        return self._path

    @property
    def input_shape(self):
        if self._gray:
            return (self._input_size[0], self._input_size[1], 1)
        else:
            return (self._input_size[0], self._input_size[1], 3)

    @property
    def n_outputs(self):
        if self._type == ClassificationType.NUMBERS:
            return 100
        elif self._type == ClassificationType.BINARY:
            return 1
        else:
            raise NotImplementedError

    def set_params(self, input_size=None, gray=None, lr=None,
                   batch_size=None, epoch_images=None, pretrained=None):
        if input_size is not None:
            self._input_size = input_size
        if gray is not None:
            self._gray = gray
        if pretrained is not None:
            self._pretrained = pretrained
        if epoch_images is not None:
            self._epoch_images = epoch_images
        if batch_size is not None:
            self._batch_size = batch_size
        if lr is not None:
            self._lr = lr

    def save(self):
        print('Save model to {}'.format(self.path))
        self._model.save(self.path)

    def _load_weights(self):
        if self._pretrained is not None:
            print("Load weights by path: {}".format(self._pretrained))
            self._model.load_weights(self._pretrained, by_name=True)

    def _save_params(self, train_dset, epochs, freeze_base, test_dset):
        params = {}
        params['base_name'] = self._base_name
        params['name'] = self._name
        params['epoch_images'] = self._epoch_images
        params['batch_size'] = self._batch_size
        params['lr'] = self._lr
        params['epochs'] = epochs
        params['train_data'] = train_dset.name
        params['test_data'] = test_dset.name if test_dset is not None else None
        params['input_shape'] = self.input_shape
        params['pretrained'] = self._pretrained
        params['freeze'] = freeze_base
        params['_model'] = str(self._model.to_json())
        f_path = osp.join(MODEL_DATA_DIR, '{}_params.txt'.format(self.name))
        with open(f_path, 'w') as fout:
            json.dump(params, fout, indent=4, sort_keys=True) 
         
    def _get_checkpointer(self, period):
        if self.type == ClassificationType.NUMBERS:
            checkpoint_path = self.name + '_weights.{epoch:02d}-{val_categorical_accuracy:.2f}-{val_loss:.2f}.hdf5'
        else:
            checkpoint_path = self.name + '_weights.{epoch:02d}-{val_binary_accuracy:.2f}-{val_loss:.2f}.hdf5'
        checkpoint_path = osp.join(MODEL_DATA_DIR, 'checkpoints', checkpoint_path)
        return ModelCheckpoint(filepath=checkpoint_path,
                               save_best_only=True, 
                                                       monitor='val_loss',
                                                       verbose=1,
                                                       period=period)

    def _get_tensorboard(self):
        log_dir = osp.join(MODEL_DATA_DIR, 'logs')
        return TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=True)

    def _get_logger(self):
        log_path = self.name + '_log.csv'
        log_path = osp.join(MODEL_DATA_DIR, 'logs', log_path)

        return CSVLogger(filename=log_path,
                                           append=True)


    def _get_stoper(self, min_delta=0.001, patience=6):
        return EarlyStopping(monitor='val_loss',
                             min_delta=min_delta,
                             patience=patience)

    def _get_reducer(self, factor=0.1, patience=3, min_lr=0.00001):
        return ReduceLROnPlateau(monitor='val_loss',
                                       factor=factor,
                                       verbose=1,
                                       patience=patience,
                                       min_lr=min_lr)


    def _get_train_generator(self, data):
        return get_data_generator(data=data,
                                  target_shape=self.input_shape,
                                  batch_size=self._batch_size,
                                  n_outputs=self.n_outputs,
                                  shuffle=True,
                                  rotation_range=0,
                                  width_shift_range=0,
                                  height_shift_range=0,
                                  featurewise_std_normalization=False)

    def _get_test_generator(self, data, shuffle=False):
        return get_data_generator(data=data,
                                  target_shape=self.input_shape,
                                  batch_size=self._batch_size,
                                  n_outputs=self.n_outputs,
                                  shuffle=False,
                                  featurewise_std_normalization=False)

    def _freeze_base_model(self):
        pass

    def _unfreeze_base_model(self):
        pass

    def train(self, train_dset, epochs, freeze_base=0, mult_lr=1, test_dset=None):
        print("TRAINING....")

        self._prepare_model()
        self._load_weights()
        self._model.summary()
        self._save_params(train_dset, epochs, freeze_base, test_dset)


        callbacks = [self._get_checkpointer(10), self._get_logger(),
                     self._get_reducer(), self._get_stoper()]

        train_generator = self._get_train_generator(train_dset.get_train(self.input_shape))
        valid_generator = self._get_train_generator(train_dset.get_test(self.input_shape))

        if freeze_base != 0:
            self._freeze_base_model()
            self._fit_step(self._lr, int(epochs * freeze_base), train_generator, valid_generator, callbacks)
            self.save()

        self._unfreeze_base_model()
        self._fit_step(self._lr * mult_lr, int(epochs * (1 - freeze_base)), train_generator, valid_generator, callbacks)
        self.save()

        if test_dset is not None:
            evaluate_model(model_path=self.path, dset=test_dset, count_images=837, batch_size=self._batch_size)

    def _fit_step(self, lr, epochs, train_generator, valid_generator, callbacks):
        compile_model(self._model, lr)

        self._model.fit_generator(train_generator,
                                  epochs=epochs, verbose=1,
                                  steps_per_epoch=self._epoch_images / self._batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=int(0.1 * self._epoch_images) / self._batch_size,
                                  callbacks=callbacks)



class VGG16Model(BaseModel):
    def __init__(self, type):
        super(VGG16Model, self).__init__('vgg16', type)
    
    def _pop_layer(self, model):
        if not model.outputs:
            raise Exception('Sequential model cannot be popped: model is empty.')
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False
   
    def _prepare_model(self):
        if self._model is not None:
            return

        self._base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self._pop_layer(self._base_model)
        self._pop_layer(self._base_model)
        self._pop_layer(self._base_model)
        self._pop_layer(self._base_model)
        
        #self._pop_layer(self._base_model)
        #self._pop_layer(self._base_model)
        #self._pop_layer(self._base_model)
        #self._pop_layer(self._base_model)

        #self._pop_layer(self._base_model)
        #self._pop_layer(self._base_model)
        #self._pop_layer(self._base_model)

        x = self._base_model.layers[-1].output#self._base_model.output

        #x = Flatten(name='vgg16_flat')(x)
        x = GlobalAveragePooling2D(name='vgg16_gap1')(x)
        x = BatchNormalization(name='vgg16_bn1')(x)
        x = Dense(128, activation='relu',
                  #kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                  kernel_regularizer=regularizers.l2(0.01),
                  name='vgg16_dense1')(x)
        x = BatchNormalization(name='vgg16_bn2')(x)
        #x = Dropout(0.5, name='vgg16_drop1')(x)
        #x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        #          name = 'vgg16_dense2')(x)

        n_outputs = self.n_outputs
        predictions = Dense(n_outputs,
                            activation='softmax' if n_outputs > 1 else 'sigmoid',
                            kernel_regularizer=regularizers.l2(0.01),
                            #kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                            name='vgg16_softmax')(x)#+ self._type.value)(x)


        self._model = Model(input=self._base_model.input, output=predictions)
   
    def _freeze_base_model(self, n=4):
        print("FREEZE LAYER {}, UNFREEZ {}".format(len(self._base_model.layers)  - n, n))
        for layer in self._base_model.layers[:-n]:
            layer.trainable = False
        for layer in self._base_model.layers[-n:]:
            layer.trainable = True


    def _unfreeze_base_model(self, n=15):
        n = len(self._base_model.layers) 
        print("FREEZE LAYER {}, UNFREEZ {}".format(len(self._base_model.layers) - n, n))
        for layer in self._base_model.layers[:-n]:
            layer.trainable = False
        for layer in self._base_model.layers[-n:]:
            layer.trainable = True

    def train(self, train_dset,  epochs, freeze_base=0,  test_dset=None):
        super(VGG16Model, self).train(train_dset, epochs, freeze_base, 
                                      mult_lr=0.1, 
                                      test_dset=test_dset)

class GerkeModel(BaseModel):
    def __init__(self, type):
        super(GerkeModel, self).__init__('gerke', type)

    def _prepare_model(self):
        if self._model is not None:
            return

        input = Input(shape=self.input_shape)
        x = input
        x = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same',
                   activation='tanh', #kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   name='gerke_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                         name='gerke_max1')(x)

        x = Conv2D(filters=30, kernel_size=(7, 7), strides=(1, 1), padding='same',
                   activation='tanh', #kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   name='gerke_conv2')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same',
                         name='gerke_max2')(x)

        x = Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   activation='tanh', # kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   name='gerke_conv3')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same',
                         name='gerke_max3')(x)

        x = Flatten(name='gerke_flat')(x)
        x = BatchNormalization(name='gerke_bn1')(x)
        x = Dense(100, activation='tanh', #kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                  kernel_regularizer=regularizers.l2(0.01),
                  name='gerke_dense1')(x)
        x = BatchNormalization(name='gerke_bn2')(x)
        #x = Dense(50, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        #          name='gerke_dense2')(x)

        predictions = Dense(self.n_outputs,
                  activation='softmax' if self.n_outputs > 1 else 'sigmoid',
                  #kernel_initializer=#RandomNormal(mean=0.0, stddev=0.01),
                  name='gerke_softmax_'+ self._type.value)(x)

        self._model = Model(input=input, output=predictions)

        if self._pretrained is not None:
            self._model.load_weights(self._pretrained, by_name=True)


class ModelType(Enum):
    VGG16 = 'vgg16'
    Gerke = 'gerke'


models = {ModelType.VGG16: VGG16Model,
          ModelType.Gerke: GerkeModel}
