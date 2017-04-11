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
ImageFile.LOAD_TRUNCATED_IMAGES = True

b_outputs = 1
# input image dimensions
img_rows, img_cols = 128, 64
#Images are RGB.
img_channels = 3

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channels))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name='d1')(x)
x = Dropout(0.5, name='dr1')(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name='d2')(x)
predictions = Dense(1, activation='sigmoid', name='s')(x)

model = Model(input=base_model.input, output=predictions)
model.summary()
model.load_weights('model/checkpoints/ftvgg16_2classes/weights.24-0.06.hdf5', by_name=False)
model.save_weights('model/checkpoints/ftvgg16_2classes/weights.24-0.06.hdf5_named')

