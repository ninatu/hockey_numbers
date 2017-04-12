#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import argparse
import sys
import cv2
from constants import VIDEO_DIR, VIDEO_FRAME_SIZE, TEMPLATE_VIDEO
from constants import BLOB_MIN_HEIGHT, BLOB_MAX_HEIGHT, BLOB_MIN_WIDTH, BLOB_MAX_WIDTH
from constants import FIELD_Y, FIELD_H, FIELD_X, FIELD_W
from utils.marking import BaseMarkingCreator

def save_frames(frame_numbers, path_to_save):
    frame_numbers = set(frame_numbers)
    start_frame = int(min(frame_numbers) / VIDEO_FRAME_SIZE) * VIDEO_FRAME_SIZE

    while(True):
        in_video = osp.join(VIDEO_DIR, \
                TEMPLATE_VIDEO.format(start_frame, start_frame + VIDEO_FRAME_SIZE - 1))
        reader = cv2.VideoCapture(in_video)
        for i_frame in range(start_frame, start_frame + VIDEO_FRAME_SIZE):
            ret, frame = reader.read()
            if not ret:
                break
            if i_frame in frame_numbers:
                cv2.imwrite(osp.join(out_folder, out_template.format(i_frame)), frame)
                frame_numbers.remove(i_frame)
        reader.release()
        if len(frame_numbers) == 0:
            break
        start_frame = int(min(frame_numbers) / VIDEO_FRAME_SIZE) * VIDEO_FRAME_SIZE

def save_markup(frame_numbers, path_to_save):
    marking = BaseMarkingCreator()
    marking.addFrames(frame_numbers)
    marking.filterBySize(BLOB_MIN_HEIGHT, BLOB_MAX_HEIGHT, BLOB_MIN_WIDTH, BLOB_MAX_WIDTH)
    marking.filterByField(FIELD_Y, FIELD_H, FIELD_X, FIELD_W)
    marking.saveAsJson(open(path_to_save, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for marking')
    parser.add_argument('frameNumbers', required=True, type=int, nargs='+', help='frame numbers')
    parser.add_argument('--framePath', type=str, nargs='?', help='dir to save frames')
    parser.add_argument('--markPath', type=str, nargs='?', help='path to save marking') 
    args = parser.parse_args()
    
    frameNumbers = args.frameNumbers
    if args.framePath:
        save_frames(frameNumbers, args.framePath)
    if args.markPath:
        save_marking(frameNumbers, args.markPath)

    """ 
    numb_start = args.first
    step = args.step
    count = args.count
    numb_end = numb_start + count * step

    templateImages = 'cska_akbars_{:d}.jpg'
    pathOut = '/media/nina/Seagate Backup Plus Drive/hockey/marking/sourceForMarking/part{}_end{}_step{}'
    pathOut = pathOut.format(numb_start, numb_end, step)
    pathOutImgs = osp.join(pathOut, 'imgs')
    pathOutMarking = osp.join(pathOut, 'marking.json')

    def ensureDir(dirname):
        if not osp.exists(dirname):
            os.makedirs(dirname)
    ensureDir(pathOut)
    ensureDir(pathOutImgs)
    save_marking(frame_numbers, )
    save_frames(numb_start, numb_end, step, pathOutImgs, templateImages)
    """ 

 
