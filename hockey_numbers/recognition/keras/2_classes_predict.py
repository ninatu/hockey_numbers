from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.initializers import RandomNormal
from PIL import ImageFile
import argparse
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("test_dir", type=str, nargs='?', help='dir with data')
parser.add_argument("count", type=int, nargs="?", help='images count')

args = parser.parse_args()
test_path = args.test_file
count = args.count

ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 256
nb_outputs = 1
nb_epoch_images = 20000
nb_epoch_images_test = int(nb_epoch_images * 0.1)

# input image dimensions
img_rows, img_cols = 128, 64
# Images are RGB.
img_channels = 3

# model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_outputs)
# model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_outputs)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channels))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
predictions = Dense(nb_outputs, activation='sigmoid')(x)

model = Model(input=base_model.input, output=predictions)
model.load_weights('model/checkpoints/ftvgg16_2classes/weights.24-0.06.hdf5_named', by_name=True)

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

test_generator = datagen.flow_from_directory(
    test_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False)

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy' if nb_outputs > 1 else 'mse',
              optimizer=sgd,
              metrics=['accuracy'])

predict = model.predict_generator(test_generator, steps=ceil(float(count)/batch_size))
test_generator.class_indices
