#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import argparse
import sys
import cv2
import tqdm
from hockey_numbers.markup.constants import VIDEO_DIR, VIDEO_FRAME_SIZE, TEMPLATE_VIDEO, TEMPLATE_FRAME
from hockey_numbers.markup.constants import BLOB_MIN_HEIGHT, BLOB_MAX_HEIGHT, BLOB_MIN_WIDTH, BLOB_MAX_WIDTH
from hockey_numbers.markup.constants import FIELD_Y, FIELD_H, FIELD_X, FIELD_W
from hockey_numbers.markup.utils.marking import BaseMarkingCreator

def save_frames(frame_numbers, dir_to_save):
    if not osp.exists(dir_to_save):
        os.mkdir(dir_to_save)
    frame_numbers = set(frame_numbers)
    video_numbers = set(map(lambda x: int(x / VIDEO_FRAME_SIZE), frame_numbers))
    for video_number in tqdm.tqdm(video_numbers):
        start_frame = video_number * VIDEO_FRAME_SIZE
        in_video = osp.join(VIDEO_DIR, \
                TEMPLATE_VIDEO.format(start_frame, start_frame + VIDEO_FRAME_SIZE - 1))
        reader = cv2.VideoCapture()
        reader.open(in_video)
        for i_frame in tqdm.tqdm(range(start_frame, start_frame + VIDEO_FRAME_SIZE)):
            ret, frame = reader.read()
            if not ret:
                break
            if i_frame in frame_numbers:
                cv2.imwrite(osp.join(dir_to_save, TEMPLATE_FRAME.format(i_frame)), frame)
        reader.release()

def save_markup(frame_numbers, path_to_save):
    marking = BaseMarkingCreator()
    marking.addFrames(frame_numbers)
    marking.filterBySize(BLOB_MIN_HEIGHT, BLOB_MAX_HEIGHT, BLOB_MIN_WIDTH, BLOB_MAX_WIDTH)
    marking.filterByField(FIELD_Y, FIELD_H, FIELD_X, FIELD_W)
    marking.saveAsJson(open(path_to_save, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for marking')
    parser.add_argument('frameNumbers', type=int, nargs='+', help='frame numbers')
    parser.add_argument('--framePath', type=str, nargs='?', help='dir to save frames')
    parser.add_argument('--markPath', type=str, nargs='?', help='path to save marking') 
    args = parser.parse_args()
    
    frameNumbers = args.frameNumbers
    if args.framePath:
        save_frames(frameNumbers, args.framePath)
    if args.markPath:
        save_markup(frameNumbers, args.markPath)

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

 
