import argparse

from datasets import DatasetType, datasets
from models import ModelType, models, ClassificationType

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


def train_model(args):
    model = models[args.model](args.ctype)
    model.set_params(input_size=(args.height, args.width), gray=args.gray)
    model.set_params(lr=args.lr, pretrained=args.weights)
    model.set_params(batch_size=args.batch_size, epoch_images=args.images_per_epoch)

    train_dset = datasets[args.dset]()
    if not train_dset.is_prepared:
        train_dset.prepare(type=args.ctype, test_per=0.2)

    if args.test_dset is not None:
        test_dset = datasets[args.test_dset]()
        if not test_dset.is_prepared:
            test_dset.prepare(type=args.ctype, test_per=1.0)
    else:
        test_dset = None

    if args.valid_dset is not None:
        valid_dset = datasets[args.test_dset]()
        if not valid_dset.is_prepared:
            valid_dset.prepare(type=args.ctype, test_per=0)
    else:
        valid_dset = None

    shape = (args.height, args.width, 1 if args.gray else 3)
    train_data = train_dset.get_train(shape)
    valid_data = valid_dset.get_train(shape) if valid_dset is not None else train_dset.get_test(shape)
    test_data = test_dset.get_test(shape) if test_dset is not None else None

    print("GET DATA: ")
    print("TRAIN DATA: path: {}, sample shape: {}".format(train_data[0], train_data[1].shape))
    print("VALID DATA: path: {}, sample shape: {}".format(valid_data[0], valid_data[1].shape))
    if test_data is not None:
        print("TEST DATA: path: {}, sample shape: {}".format(test_data[0], test_data[1].shape))

    model.start_session()
    model.train(train_data=train_data,
                valid_data=valid_data,
                epochs=args.epochs,
                freeze_base=args.freeze,
                test_data=test_data)
    model.clear_session()


def evaluate_model(args):
    model = models[args.model](args.ctype)
    model.set_params(input_size=(args.height, args.width), gray=args.gray)
    model.set_params(pretrained=args.weights)
    model.set_params(batch_size=args.batch_size)

    test_dset = datasets[args.dset]()
    if not test_dset.is_prepared:
        test_dset.prepare(type=args.ctype, test_per=1)

    shape = (args.height, args.width, 1 if args.gray else 3)
    test_data = test_dset.get_test(shape)

    model.start_session()
    model.evaluate(test_data, test_images=args.count)
    model.clear_session()


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
    train_parser.add_argument('--valid_dset', type=DatasetType, help='dataset type for valid')
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
    train_parser.set_defaults(func=train_model)

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate model')
    evaluate_parser.add_argument('model', type=ModelType, help='model type')
    evaluate_parser.add_argument('dset', type=DatasetType, help='dataset type for evaluate')
    evaluate_parser.add_argument('--weights', type=str, required=True, help='pretrainet weights')
    evaluate_parser.add_argument('--count', type=int, required=True, help='count images for evaluate')

    evaluate_parser.add_argument('--ctype', type=ClassificationType, required=False,
                              default=ClassificationType.NUMBERS, help='classification type')
    evaluate_parser.add_argument('-w', '--width', type=int, required=False, default=DEFAULT_W, help='width data')
    evaluate_parser.add_argument('-r', '--height', type=int, required=False, default=DEFAULT_H, help='height data')
    evaluate_parser.add_argument('--gray', action='store_true', default=False, help='convert to grayscale')
    evaluate_parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='batch_size')
    evaluate_parser.set_defaults(func=evaluate_model)


    args = parser.parse_args()
    args.func(args)


if __name__=='__main__':
    main()
