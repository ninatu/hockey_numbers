from utils.markup import Markup
from utils.image_process import crop_by_rect
from utils.load import load_frame, load_mask

import json
import os.path as osp
import argparse
import h5py
import re
import tqdm


def print_statistics(args):
    markup = Markup()
    markup.merge(args.markup_file)
    statistics = markup.get_statistics()

    out_file = osp.join(osp.dirname(args.markup_file), 'statistics.txt')
    with open(out_file, 'w') as f_out:
        json.dump(statistics, f_out)

    print(statistics)


def merge_markup(args):
    markup = Markup()
    for file in args.files:
        markup.merge(file)

    with open(args.output, 'w') as f_out:
        json.dump(markup.to_json(is_annotation=True), f_out)


def save_by_mark(args):
    markup_file = args.markup_file
    mark = args.mark
    out_name = args.out_name
    save_format = args.save_format
    with_masks = not args.not_masks
    distibute = args.distribute

    assert save_format == 'dir' or save_format == 'hdf5'

    markup = Markup()
    markup.merge(markup_file)
    marked_frame = markup.get_by_mark(mark)

    if save_format=='dir':
        pass

    if save_format=='hdf5':
        out_db = h5py.File(out_name, 'w')
        img_gr = out_db.create_group('image')
        if with_masks:
            mask_gr = out_db.create_group('mask')

        for frame_name, frame in tqdm.tqdm(marked_frame.items()):
            numb = int(re.search(r'\d+', frame_name).group(0))

            img_frame = load_frame(numb)
            if with_masks:
                img_mask = load_mask(numb)

            for obj in frame.objects:
                img = crop_by_rect(img_frame, obj.x, obj.y, obj.w, obj.h)
                img_name = obj.data['img_name'] # TODO
                img_dset = img_gr.create_dataset(img_name, img.shape, '|u1', img)
                img_dset.attrs[mark] = obj.data[mark]

                if with_masks:
                    mask = crop_by_rect(img_mask, obj.x, obj.y, obj.w, obj.h)
                    img_dset = mask_gr.create_dataset(img_name, mask.shape, '|u1', mask)
                    img_dset.attrs[mark] = obj.data[mark]

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_merge = subparsers.add_parser('merge', help='merge marking files')
    parser_merge.add_argument('files', type=str, nargs='*', help='files to merge')
    parser_merge.add_argument('-o', '--output', type=str, required=True, help='file to save')
    parser_merge.set_defaults(func=merge_markup)

    parser_print = subparsers.add_parser('print_statistics', help='print markup statistics')
    parser_print.add_argument('markup_file', nargs='?', help='input file with markup')
    parser_print.set_defaults(func=print_statistics)

    parser_save = subparsers.add_parser('save', help='save blobs in marking file by mark')
    parser_save.add_argument('markup_file', type=str, nargs='?', help='markup file')
    parser_save.add_argument('mark', type=str, nargs='?', help='mark')
    parser_save.add_argument('out_name', type=str, nargs='?', help='output name file or dir to save')
    parser_save.add_argument('--save_format', type=str, nargs='?', default='hdf5', help='hdf5 or dir, default=hdf5')
    parser_save.add_argument('--not_masks', action='store_true', help='not save masks')
    parser_save.add_argument('--distribute', action='store_true',
                            help='distribute by subdirs, only for save_format=dir anf mark=number')
    parser_save.set_defaults(func=save_by_mark)


    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
