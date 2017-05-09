import json
import numpy as np

from keras.models import load_model
from keras import backend as K
import tensorflow as tf

from model_utils import get_data_generator, compile_model


def start_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


def clear_session():
    K.clear_session()


def evaluate_model(model_path, dset, count_images, batch_size):
    print("Evaluation model on {} dataset".format(dset.name))
    model = load_model(model_path)

    input_shape = model.input_shape[1:]
    n_outputs = model.output_shape[1]
    print("Input shape images: {}\nClassification on {} classes".format(input_shape, n_outputs))

    data_generator = get_data_generator(data=dset.get_test(input_shape),
                                        target_shape=input_shape,
                                        batch_size=batch_size,
                                        n_outputs=n_outputs)
    compile_model(model, lr=0.1)
    scores = model.evaluate_generator(data_generator,
                                      steps=count_images / batch_size)

    for metric, score in zip(model.metrics_names, scores):
        print("{} = {}".format(metric, score))


def predict_model(model_path, dset, count_images, batch_size, path_to_save):
    print("Predict model on {} dataset".format(dset.name))

    model = load_model(model_path)
    input_shape = model.input_shape[1:]
    n_outputs = model.output_shape[1]

    print("Input shape images: {}\nClassification on {} classes".format(input_shape, n_outputs))

    data_generator = get_data_generator(data=dset.get_test(input_shape),
                                        target_shape=input_shape,
                                        batch_size=batch_size,
                                        n_outputs=n_outputs)
    compile_model(model, lr=0.1)

    predict = model.predict_generator(data_generator,
                                      steps=np.ceil(float(count_images) / batch_size))
    ans = dict()
    ans['class_indices'] = data_generator.class_indices
    ans['filenames'] = data_generator.filenames
    ans['predict'] = predict.tolist()
    with open(path_to_save, 'w') as fout:
        json.dump(ans, fout)
