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

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, verbose=1, min_lr=1e-7)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
csv_logger = CSVLogger('./model/ft_100class.csv')
checkpoint_step1 = ModelCheckpoint('./model/checkpoints/ft_100class/weights_step1.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
checkpoint_step2 = ModelCheckpoint('./model/checkpoints/ft_100class/weights_step2.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
#tensorboard = None#TensorBoard(log_dir='./model/logs', histogram_freq=0, write_graph=False, write_images=True)

#callbacks = [lr_reducer, early_stopper, csv_logger, checkpoint]
#callbacks = [c for c in callbacks if c is not None]

batch_size = 256
nb_outputs = 100
nb_epoch_step1 = 100
nb_epoch_step2 = 100
nb_epoch_images = 20000
nb_epoch_images_test = int(nb_epoch_images * 0.1)

# input image dimensions
img_rows, img_cols = 128, 64
#Images are RGB.
img_channels = 3

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_rows, img_cols, img_channels))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name='d1')(x)
x = Dropout(0.5, name='dr1')(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name='d2')(x)
prediction = Dense(nb_outputs, activation='softmax', name='softmax')(x)

model = Model(input=base_model.input, output=prediction)
model.load_weights('model/checkpoints/ftvgg16_2classes/weights.24-0.06.hdf5_named', by_name=True)

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.04,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.04,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

train_generator = datagen.flow_from_directory(
        'images/data/train',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="categorical")

test_generator = datagen.flow_from_directory(
        'images/data/test',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="categorical")

# step1
for layer in base_model.layers:
    layer.trainable = False

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy' if nb_outputs > 1 else 'mse',
              optimizer=sgd,
              metrics=['accuracy'])

# Fit the model on the batches generated by datagen.flow().
callbacks = [lr_reducer, early_stopper, csv_logger, checkpoint_step1]
model.fit_generator(train_generator,
                    epochs=nb_epoch_step1, verbose=1,
                    steps_per_epoch=nb_epoch_images / batch_size,
                    validation_data=test_generator,
                    validation_steps=nb_epoch_images_test / batch_size,
                    callbacks=callbacks)

model.save("model/model_ft_100class_1.h5" )

# step2
for layer in base_model.layers:
    layer.trainable = True

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy' if nb_outputs > 1 else 'mse',
              optimizer=sgd,
              metrics=['accuracy'])

callbacks = [lr_reducer, early_stopper, csv_logger, checkpoint_step2]
model.fit_generator(train_generator,
                    epochs=nb_epoch_step2, verbose=1,
                    steps_per_epoch=nb_epoch_images / batch_size,
                    validation_data=test_generator,
                    validation_steps=nb_epoch_images_test / batch_size,
                    callbacks=callbacks)

model.save("model/model_ft_100class_2.h5" )

