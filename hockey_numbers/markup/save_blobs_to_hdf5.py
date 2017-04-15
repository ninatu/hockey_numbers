#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Find and Save crops from frames"""

import argparse
import h5py
import tqdm
from hockey_numbers.markup.utils.load import load_frame, load_mask
from hockey_numbers.markup.utils.blob import getBlobsFromMasks, filterBlobsBySize, filterBlobsByField
from hockey_numbers.markup.utils.image_process import cropByRect, md5_hash
from hockey_numbers.markup.constants import TEMPLATE_IMAGE


def save_blobs_to_hdf5(frame_numbers, outfile, filtered=True):
    """Find and Save crops from frames"""

    out_db = h5py.File(outfile, 'w')
    out_db.create_group('/image')
    out_db.create_group('/mask')
    yet_saved = set()

    for numb in tqdm.tqdm(frame_numbers):
        frame = load_frame(numb)
        mask = load_mask(numb)
        blobs = getBlobsFromMasks(mask)

        if filtered:
            blobs = filterBlobsBySize(blobs)
            blobs = filterBlobsByField(blobs)

        for blob in blobs:
            x, y, w, h = blob.x, blob.y, blob.width, blob.height
            img = cropByRect(frame, x, y, w, h)
            img_mask = cropByRect(mask, x, y, w, h)

            img_name = TEMPLATE_IMAGE.format(md5_hash(img))
            if img_name in yet_saved:
                continue

            img_dset = out_db['image'].create_dataset(img_name, img.shape, '|u1', img)
            img_dset.attrs['frame_number'] = numb
            img_dset.attrs['x'] = x
            img_dset.attrs['y'] = y
            img_dset.attrs['w'] = w
            img_dset.attrs['h'] = h
            out_db['mask'].create_dataset(img_name, img_mask.shape, '|u1', img_mask)
            yet_saved.add(img_name)

def main():
    """Parse args and run function"""

    parser = argparse.ArgumentParser(description='Find and Save crops from frames')
    parser.add_argument('frameNumbers', type=int, nargs='+', help='frame numbers')
    parser.add_argument('-o', '--outfile', type=str, required=True, nargs='?', help='file to save hdf5')
    parser.add_argument('--filter', action='store_true', default=False,
                        help='flag for filtering blobs by size and field')
    args = parser.parse_args()
    save_blobs_to_hdf5(args.frameNumbers, args.outfile, args.filter)


if __name__=='__main__':
    main()