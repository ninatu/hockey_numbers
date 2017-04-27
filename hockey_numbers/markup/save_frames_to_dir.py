#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Capture frames video save its to dir"""

import os
import os.path as osp
import sys
import argparse
import tqdm
import cv2
import shutil

from constants import VIDEO_DIR, VIDEO_FRAME_SIZE, TEMPLATE_VIDEO, TEMPLATE_FRAME, FRAME_DIR


def copy_frames(frame_numbers, dir_to_save):
    print("Copying {} frames...".format(len(frame_numbers)))

    for frame_number in tqdm.tqdm(frame_numbers):
        frame_name = TEMPLATE_FRAME.format(frame_number)
        in_path = os.path.join(FRAME_DIR, frame_name)

        if not os.path.exists(in_path):
            print("Frame {} is not saved yet!".format(frame_name))
            continue

        shutil.copy(in_path, dir_to_save)


def capture_frames(frame_numbers):
    print("Capturing {} frames...".format(len(frame_numbers)))

    frame_numbers = set(frame_numbers)
    video_numbers = set(map(lambda x: int(x / VIDEO_FRAME_SIZE), frame_numbers))
    for video_number in tqdm.tqdm(video_numbers):
        start_frame = video_number * VIDEO_FRAME_SIZE
        in_video = osp.join(VIDEO_DIR, \
                            TEMPLATE_VIDEO.format(start_frame, start_frame + VIDEO_FRAME_SIZE - 1))
        reader = cv2.VideoCapture()
        reader.open(in_video)
        for i_frame in range(start_frame, start_frame + VIDEO_FRAME_SIZE):
            ret, frame = reader.read()
            if not ret:
                break
            if i_frame in frame_numbers:
                cv2.imwrite(osp.join(FRAME_DIR, TEMPLATE_FRAME.format(i_frame)), frame)

        reader.release()


def save_frames_to_dir(frame_numbers, dir_to_save):
    """Capture frames video save its to dir"""

    if not osp.exists(dir_to_save):
        os.mkdir(dir_to_save)
    frame_numbers = set(frame_numbers)

    not_captured = []

    for frame_number in frame_numbers:
        frame_path = TEMPLATE_FRAME.format(frame_number)
        frame_path = os.path.join(FRAME_DIR, frame_path)

        if not os.path.exists(frame_path):
            not_captured.append(frame_number)

    capture_frames(not_captured)
    if not os.path.samefile(FRAME_DIR, dir_to_save):
        copy_frames(frame_numbers, dir_to_save)



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