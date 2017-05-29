#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tool for train, test, predict neural networks"""

from datasets import DatasetType, datasets
from models import ModelType, models, ClassificationType
from api import start_session, clear_session, evaluate_model, predict_model

import argparse


DEFAULT_TEST = 0.2
DEFAULT_H = 200
DEFAULT_W = 100
DEFAULT_EPOCHS = 100
DEFAULT_EPOCH_IMAGES = 27000
DEFAULT_FREEZE_BASE_EPOCH = 0
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.1


def dset_prepare(args):
    dset = datasets[args.dset]()
    dset.prepare(type=args.ctype, test_per=args.t)


def train(args):
    model = models[args.model](args.ctype)
    model.set_params(input_size=(args.height, args.width), gray=args.gray)
    model.set_params(lr=args.lr, pretrained=args.weights)
    model.set_params(batch_size=args.batch_size, epoch_images=args.images_per_epoch)

    train_dset = datasets[args.dset]()
    if not train_dset.is_prepared:
        raise "Train dataset not prepared!"

    if args.test_dset is not None:
        test_dset = datasets[args.test_dset]()
        if not test_dset.is_prepared:
            raise "Test dataset not prepared!"
    else:
        test_dset = None

    start_session()
    model.train(train_dset=train_dset,
                epochs=args.epochs,
                freeze_base=args.freeze,
                test_dset=test_dset)
    clear_session()


def evaluate(args):
    test_dset = datasets[args.dset]()
    if not test_dset.is_prepared:
        raise "Dataset not prepared!"

    start_session()
    evaluate_model(args.model, test_dset, args.count, args.batch_size)
    clear_session()


def predict(args):
    test_dset = datasets[args.dset]()
    if not test_dset.is_prepared:
        raise "Dataset not prepared!"

    start_session()
    predict_model(args.model, test_dset, args.count, args.batch_size, args.outfile)
    clear_session()
    

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title='actions')

    dset_parser = subparsers.add_parser('prepare', help='prepare dataset for train and test')
    dset_parser.add_argument('dset', type=DatasetType, help='dataset type')
    dset_parser.add_argument('--ctype', type=ClassificationType, default=ClassificationType.NUMBERS,
                             help='prepare for classification type')
    dset_parser.add_argument('-t', type=float, required=False, default=DEFAULT_TEST, help='percent of test part')
    dset_parser.set_defaults(func=dset_prepare)

    train_parser = subparsers.add_parser('train', help='train model')
    train_parser.add_argument('model', type=ModelType, help='model type')
    train_parser.add_argument('dset', type=DatasetType, help='dataset type for train')
    train_parser.add_argument('--ctype', type=ClassificationType, required=False,
                              default=ClassificationType.NUMBERS, help='classification type')
    train_parser.add_argument('-w', '--width', type=int, required=False, default=DEFAULT_W, help='width data')
    train_parser.add_argument('-r', '--height', type=int, required=False, default=DEFAULT_H, help='height data')
    train_parser.add_argument('--gray', action='store_true', default=False, help='convert to grayscale')
    train_parser.add_argument('--lr', type=float, required=False, default=DEFAULT_LR, help='learning rate')
    train_parser.add_argument('-e', '--epochs', type=int, default=DEFAULT_EPOCHS, help='count train epoch')
    train_parser.add_argument('--images_per_epoch', type=int, default=DEFAULT_EPOCH_IMAGES, help='images per one epoch')
    train_parser.add_argument('--freeze', type=float, default=DEFAULT_FREEZE_BASE_EPOCH,
                              help='persent of count epoch with freeze base model layer')
    train_parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='batch_size')
    train_parser.add_argument('--test_dset', type=DatasetType, default=None, help='dataset type for test')
    train_parser.add_argument('--weights', type=str, default=None, help='pretrainet weights')
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate model')
    evaluate_parser.add_argument('--dset', type=DatasetType, required=True, help='dataset type for evaluate')
    evaluate_parser.add_argument('--model', type=str, required=True, help='path to pretrained model')
    evaluate_parser.add_argument('--count', type=int, required=True, help='count images for evaluate')
    evaluate_parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='batch_size')
    evaluate_parser.set_defaults(func=evaluate)

    predict_parser = subparsers.add_parser('predict', help='predict model')
    predict_parser.add_argument('--dset', type=DatasetType, required=True, help='dataset type for predict')
    predict_parser.add_argument('--model', type=str, required=True, help='path to pretrained model')
    predict_parser.add_argument('--count', type=int, required=True, help='count images in dataset')
    predict_parser.add_argument('-o', '--outfile', type=str, required=True, help='file to save results')
    predict_parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='batch_size')
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__=='__main__':
    main()
