#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Capture frames video save its to dir"""

import os
import os.path as osp
import sys
import argparse
import tqdm
import cv2

from hockey_numbers.markup.constants import VIDEO_DIR, VIDEO_FRAME_SIZE, TEMPLATE_VIDEO, TEMPLATE_FRAME, FRAME_DIR


def save_frames_to_dir(frame_numbers, dir_to_save):
    """Capture frames video save its to dir"""

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

def main():
    """Parse args and run function"""

    parser = argparse.ArgumentParser(description='Find and Save crops from frames')
    parser.add_argument('frameNumbers', type=int, nargs='+', help='frame numbers')
    parser.add_argument('-o', '--outdir', type=str, default=FRAME_DIR, nargs='?',
                        help='dir to save frames, default=FRAME_DIR')
    args = parser.parse_args()
    save_frames_to_dir(args.frameNumbers, args.outdir)

if __name__ == '__main__':
    main()