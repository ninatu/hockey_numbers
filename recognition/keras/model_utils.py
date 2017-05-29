#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


def get_data_generator(data, target_shape, batch_size, n_outputs, shuffle=False,
                       rotation_range=0, width_shift_range=0, height_shift_range=0,
                       featurewise_std_normalization=False):
    generator = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=featurewise_std_normalization,
                                   rotation_range=rotation_range,
                                   width_shift_range=width_shift_range,
                                   height_shift_range=height_shift_range)
    data_dir, data_sample = data
    generator.fit(data_sample)

    return generator.flow_from_directory(data_dir,
                                         target_size=(target_shape[0], target_shape[1]),
                                         batch_size=batch_size,
                                         class_mode='binary' if n_outputs == 1 else "categorical",
                                         shuffle=shuffle,
                                         color_mode="grayscale" if target_shape[2] == 1 else 'rgb')


def compile_model(model, lr):
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    n_outputs = model.output_shape[1]
    model.compile(loss='categorical_crossentropy' if n_outputs > 1 else 'binary_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy' if n_outputs > 1 else 'binary_accuracy'])