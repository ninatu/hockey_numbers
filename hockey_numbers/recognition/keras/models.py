from abc import abstractmethod
from abc import ABCMeta
from enum import Enum
import datetime
import os.path as osp
import numpy as np
import sys

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
from keras import backend as K
import tensorflow as tf

MODEL_DATA_DIR = 'models'


class ClassificationType(Enum):
    NUMBERS = 'numbers'
    BINARY = 'binary'


class AbstractModel(metaclass=ABCMeta):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
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

        self._model = None
        self._base_model = None

        self._input_size = (128, 64)
        self._gray = False
        self._pretrained = None
        self._lr = 0.1
        self._batch_size = 256
        self._epoch_images = 1000

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
        assert self._model is not None
        self._model.save(osp.join(MODEL_DATA_DIR, self._base_name + '_model.hdf5'))
        self._model.save(osp.join(MODEL_DATA_DIR, self.name + '_model.hdf5'))

    def _save_test_results(self, dset_name, scores):
        f_path = osp.join(MODEL_DATA_DIR, '{}_test_{}_acc_{}.txt'.format(self.name, dset_name, scores[1]))
        with open(f_path, 'w') as fout:
            for metric, score in zip(self._model.metrics_names, scores):
                fout.write("{} = {}\n".format(metric, score))

    def _get_checkpointer(self, period):
        if self.type == ClassificationType.NUMBERS:
            checkpoint_path = self.name + '_weights.{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5'
        else:
            checkpoint_path = self.name + '_weights.{epoch:02d}-{val_binary_accuracy:.2f}-{val_loss:.2f}.hdf5'
        checkpoint_path = osp.join(MODEL_DATA_DIR, 'checkpoints', checkpoint_path)
        return ModelCheckpoint(filepath=checkpoint_path,
                                                       monitor='val_loss',
                                                       verbose=1,
                                                       period=period)

    def _get_logger(self):
        log_path = self.name + '_log.csv'
        log_path = osp.join(MODEL_DATA_DIR, 'logs', log_path)

        return CSVLogger(filename=log_path,
                                           append=True)

    def _get_stoper(self, min_delta=0.001, patience=10):
        return EarlyStopping(monitor='val_loss',
                             min_delta=min_delta,
                             patience=patience)


    def _get_reducer(self, factor=0.1, patience=5, min_lr=0.00001):
        return ReduceLROnPlateau(monitor='val_loss',
                                       factor=factor,
                                       verbose=1,
                                       patience=patience,
                                       min_lr=min_lr)

    def _get_train_generator(self):
        return  ImageDataGenerator(featurewise_center=True,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   rotation_range=0,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   fill_mode='nearest')

    def _get_test_generator(self):
        return ImageDataGenerator()

    def clear_session(self):
        K.clear_session()

    def start_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)
    
    def _freeze_base_model(self):
        if self._base_model is not None:
            for layer in self._base_model.layers:
                layer.trainable = False

    def _unfreeze_base_model(self):
        if self._base_model is not None:
            for layer in self._base_model.layers:
                layer.trainable = True

    def train(self, train_dir, valid_dir, epochs, freeze_base=0, numpy_data_sample=None, test_dir=None):

        self._prepare_model()

        self._model.summary()

        callbacks = [self._get_checkpointer(10), self._get_logger(),
                     self._get_reducer(), self._get_stoper()]

        generator = self._get_train_generator()
        if  numpy_data_sample is not None:
            generator.fit(numpy_data_sample)

        train_generator = generator.flow_from_directory(train_dir,
                                                        target_size=self._input_size,
                                                        batch_size=self._batch_size,
                                                        class_mode='binary' if self.n_outputs == 1 else "categorical",
                                                        shuffle=True,
                                                        color_mode="grayscale" if self._gray else 'rgb')
                                                        #save_to_dir='data/gen_img',
                                                        #save_prefix='img',
                                                        #save_format='png')

        valid_generator = generator.flow_from_directory(valid_dir,
                                                        target_size=self._input_size,
                                                        batch_size=self._batch_size,
                                                        class_mode='binary' if self.n_outputs == 1 else "categorical",
                                                        shuffle=True,
                                                        color_mode="grayscale" if self._gray else 'rgb')

        self._freeze_base_model()
        self._fit_step(self._lr, int(epochs * freeze_base), train_generator, valid_generator, callbacks)

        self._unfreeze_base_model()
        self._fit_step(self._lr, int(epochs - (1 - freeze_base)), train_generator, valid_generator, callbacks)
        self.save()

        if test_dir is not None:
            self.evaluate(test_dir)

    def _compile(self, lr=0.01):
        sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

        self._model.compile(loss='categorical_crossentropy' if self.n_outputs > 1 else 'binary_crossentropy',
                            optimizer=sgd,
                            metrics=['categorical_accuracy' if self.n_outputs > 1 else 'binary_accuracy'])

    def _fit_step(self, lr, epochs, train_generator, valid_generator, callbacks):

        self._compile(lr)

        self._model.fit_generator(train_generator,
                                  epochs=epochs, verbose=1,
                                  steps_per_epoch=self._epoch_images / self._batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=int(0.1 * self._epoch_images) / self._batch_size,
                                  callbacks=callbacks)

    def evaluate(self, test_dir, test_images=10000):
        self._compile()

        generator = self._get_test_generator()
        test_generator = generator.flow_from_directory(test_dir,
                                                       target_size=self._input_size,
                                                       batch_size=self._batch_size,
                                                       class_mode='binary' if self.n_outputs == 1 else "categorical",
                                                       shuffle=True,
                                                       color_mode="grayscale" if self._gray else 'rgb')
        scores = self._model.evaluate_generator(test_generator,
                                       steps=test_images/self._batch_size)

        for metric, score in zip(self._model.metrics_names, scores):
            print("{} = {}".format(metric, score))

    def predict(self, img):
        pass
        #self._save_test_results(dset.name, scores)





class VGG16Model(BaseModel):
    def __init__(self, type):
        super(VGG16Model, self).__init__('vgg16', type)

    def _prepare_model(self):
        if self._model is not None:
            return

        self._base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = self._base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                  name='vgg16_dense1')(x)
        #x = Dropout(0.7, name='vgg16_drop1')(x)
        #x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        #          name = 'vgg16_dense2')(x)

        n_outputs = self.n_outputs
        predictions = Dense(n_outputs,
                            activation='softmax' if n_outputs > 1 else 'sigmoid', name='vgg16_softmax',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)

        self._model = Model(input=self._base_model.input, output=predictions)

        if self._pretrained is not None:
            self._model.load_weights(self._pretrained, by_name=True)




class GerkeModel(BaseModel):
    def __init__(self, type):
        super(GerkeModel, self).__init__('gerke', type)

    def _prepare_model(self):
        if self._model is not None:
            return

        input = Input(shape=self.input_shape)
        x = input
        x = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same',
                   activation='relu', #kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   name='gerke_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                         name='gerke_max1')(x)

        x = Conv2D(filters=30, kernel_size=(7, 7), strides=(1, 1), padding='same',
                   activation='relu', #kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   name='gerke_conv2')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same',
                         name='gerke_max2')(x)

        x = Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   activation='relu', # kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   name='gerke_conv3')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same',
                         name='gerke_max3')(x)

        x = Flatten(name='gerke_flat')(x)
        x = Dense(50, activation='relu', #kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                  name='gerke_dense1')(x)

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
