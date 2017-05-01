from abc import abstractmethod
from abc import ABCMeta
from enum import Enum
import datetime
import os.path as osp
import numpy as np
import sys

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


MODEL_DATA_DIR = 'models'


class ClassificationType(Enum):
    NUMBERS = 'numbers'
    BINARY = 'binary'


class BaseModel(metaclass=ABCMeta):
    def __init__(self, name):
        self._base_name = name
        self._name = '{}_{}'.format(name, datetime.datetime.now().strftime("%d_%m_%Y_%H_%M"))
        self._model = None
        self._input_size = (128, 64)
        self._gray = False
        self._pretrained = None
        self._lr = 0.1
        self._batch_size = 256
        self._epoch_images = 1000

    @property
    def name(self):
        return self._name

    @abstractmethod
    def train(self, train_dset, epoche, test_dset=None):
        raise NotImplementedError

    def evaluate(self, dset, test_images=800):
        generator = self._get_test_generator()
        test_generator = generator.flow_from_directory(dset.test_directory,
                                                       target_size=self._input_size,
                                                       batch_size=self._batch_size,
                                                       class_mode='binary' if self.n_outputs == 1 else "categorical",
                                                       shuffle=True,
                                                       color_mode="grayscale" if self._gray else 'rgb')
        scores = self._model.evaluate_generator(test_generator,
                                       steps=test_images/self._batch_size)

        for metric, score in zip(self._model.metrics_names, scores):
            print("{} = {}".format(metric, score))

        self._save_test_results(dset.name, scores)

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
        checkpoint_path = self.name + '_weights.{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint_path = osp.join(MODEL_DATA_DIR, 'checkpoints', checkpoint_path)
        return ModelCheckpoint(filepath=checkpoint_path,
                                                       monitor='val_acc',
                                                       verbose=1,
                                                       period=period)

    def _get_logger(self):
        log_path = self.name + '_log.csv'
        log_path = osp.join(MODEL_DATA_DIR, 'logs', log_path)

        return CSVLogger(filename=log_path,
                                           append=True)

    def _get_stoper(self, min_delta=0.001, patience=10):
        return EarlyStopping(monitor='val_acc',
                             min_delta=min_delta,
                             patience=patience)


    def _get_reducer(self, factor=np.sqrt(0.1), patience=5, min_lr=0.00001):
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
                                   rotation_range=5,
                                   width_shift_range=0.10,
                                   height_shift_range=0.10,
                                   fill_mode='nearest')
    def _get_test_generator(self):
        return ImageDataGenerator()

    def clear_session(self):
        K.clear_session()


class VGG16Model(BaseModel):
    def __init__(self, type):
        super(VGG16Model, self).__init__('vgg16_'+type.value)
        self._type = type
        self._model = None
        self._input_size = (128, 64)
        self._gray = False
        self._pretrained = None
        self._batch_size = 256
        self._lr = 0.1

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

    def _prepare_model(self):
        if self._model is not None:
            return

        self._base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = self._base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                  name='vgg16_dense1')(x)
        x = Dropout(0.5, name='vgg16_drop1')(x)
        x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.0010),
                  name = 'vgg16_dense2')(x)

        n_outputs = self.n_outputs
        predictions = Dense(n_outputs,
                            activation='softmax' if n_outputs > 1 else 'sigmoid', name='vgg16_softmax')(x)

        self._model = Model(input=self._base_model.input, output=predictions)

        if self._pretrained:
            self._model.load_weights(self._pretrained, by_name=True)

    def _freeze_base_model(self):
        for layer in self._base_model.layers:
            layer.trainable = False

    def _unfreeze_base_model(self):
        for layer in self._base_model.layers:
            layer.trainable = True

    def train(self, train_dset, epochs, test_dset=None):

        self._prepare_model()

        self._model.summary()

        callbacks = [self._get_checkpointer(10), self._get_logger(),
                     self._get_reducer(), self._get_stoper()]

        generator = self._get_train_generator()
        generator.fit(train_dset.numpy_sample(self.input_shape))

        train_generator = generator.flow_from_directory(train_dset.train_directory,
                                                        target_size=self._input_size,
                                                        batch_size=self._batch_size,
                                                        class_mode='binary' if self.n_outputs == 1 else "categorical",
                                                        shuffle=True,
                                                        color_mode="grayscale" if self._gray else 'rgb')
                                                        #save_to_dir='data/gen_img',
                                                        #save_prefix='img',
                                                        #save_format='png')

        valid_generator = generator.flow_from_directory(train_dset.test_directory,
                                                        target_size=self._input_size,
                                                        batch_size=self._batch_size,
                                                        class_mode='binary' if self.n_outputs == 1 else "categorical",
                                                        shuffle=True,
                                                        color_mode="grayscale" if self._gray else 'rgb')

        self._freeze_base_model()
        self._fit_step(self._lr, int(epochs * 0.3), train_generator, valid_generator, callbacks)

        self._unfreeze_base_model()
        self._fit_step(self._lr * 0.1, int(epochs * 0.7), train_generator, valid_generator, callbacks)
        self.save()

        if test_dset is not None:
            self.evaluate(test_dset)

    def _fit_step(self, lr, epochs, train_generator, valid_generator, callbacks):

        sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

        self._model.compile(loss='categorical_crossentropy' if self.n_outputs > 1 else 'binary_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])

        self._model.fit_generator(train_generator,
                                    epochs=epochs, verbose=1,
                                    steps_per_epoch=self._epoch_images / self._batch_size,
                                    validation_data=valid_generator,
                                    validation_steps=int(0.1 * self._epoch_images) / self._batch_size,
                                    callbacks=callbacks)






class ModelType(Enum):
    VGG16 = 'vgg16'


models = {ModelType.VGG16: VGG16Model}