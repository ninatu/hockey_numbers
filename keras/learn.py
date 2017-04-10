from keras.preprocessing.image import ImageDataGenerator

import resnet

import numpy as np

from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
csv_logger = CSVLogger('./model/resnet_log.csv')
checkpoint = ModelCheckpoint('./model/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
tensorboard = None#TensorBoard(log_dir='./model/logs', histogram_freq=0, write_graph=False, write_images=True)

callbacks = [lr_reducer, early_stopper, csv_logger, checkpoint, tensorboard]
callbacks = [c for c in callbacks if c is not None]

batch_size = 32
nb_outputs = 1
nb_epoch = 200
nb_epoch_images = 20000
nb_epoch_images_test = int(nb_epoch_images * 0.1)

# input image dimensions
img_rows, img_cols = 256, 256
#Images are RGB.
img_channels = 3

#model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_outputs)
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_outputs)
model.compile(loss='categorical_crossentropy' if nb_outputs > 1 else 'mse',
              optimizer='adam',
              metrics=['accuracy'])

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

train_generator = datagen.flow_from_directory(
        'images/data/train',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary')

test_generator = datagen.flow_from_directory(
        'images/data/test',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary')

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(train_generator,
                    epochs=nb_epoch,# verbose=1,
                    steps_per_epoch=nb_epoch_images / batch_size,
                    validation_data=test_generator,
                    validation_steps=nb_epoch_images_test / batch_size,
                    callbacks=callbacks)

model.save("model_result")