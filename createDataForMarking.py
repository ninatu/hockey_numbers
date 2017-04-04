#!/usr/bin/python3

from libs.marking import BaseMarkingCreator
import os.path as osp
import os
import argparse
import sys
import cv2

STEP_IN_FRAMES = 4

parser = argparse.ArgumentParser(description='Prepare data for marking')
parser.add_argument('-s', '--start', required=True, type=int, nargs='?', help='number of start frame')
parser.add_argument('-c', '--count', required=True, type=int, nargs='?', help='count of frames')

#saving frames
def save_frames(number_start, number_end, step, out_folder, out_template):
    video_dir = '/media/nina/Seagate Backup Plus Drive/hockey/processed_video/'
    video_template = "cska_akbars_cam_3_{:d}_{:d}.avi"

    in_video = video_dir + video_template.format(numb_start, numb_start + 1199)
    reader = cv2.VideoCapture(in_video)

    for i_frame in range(numb_start, numb_end):
            ret, frame = reader.read()
            if not ret:
                break
            print(i_frame)
            if (i_frame - numb_start) % step == 0:
                cv2.imwrite(osp.join(out_folder, out_template.format(i_frame)), frame)
    reader.release()

def save_marking(number_start, number_end, step, template_images, path_to_save):
    masks_dir = '/media/nina/Seagate Backup Plus Drive/hockey/masks/'
    template_masks = 'mask{:d}.png'

    marking = BaseMarkingCreator(masks_dir, template_masks, template_images)
    marking.addFrames(numb_start, numb_end, step=step)
    marking.filterBySize(70, 190, 40, 120)
    marking.filterByField(0, 820, 0, 5700)
    marking.saveAsJson(open(path_to_save, 'w'))


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    
    numb_start = args.start
    count = args.count
    numb_end = numb_start + count
    step = STEP_IN_FRAMES 

    templateImages = 'cska_akbars_{:d}.jpg'
    pathOut = '/media/nina/Seagate Backup Plus Drive/hockey/marking/sourceForMarking/part%d_%d'%(numb_start, numb_end)
    pathOutImgs = osp.join(pathOut, 'imgs')
    pathOutMarking = osp.join(pathOut, 'marking.json')

    def ensureDir(dirname):
        if not osp.exists(dirname):
            os.makedirs(dirname)
    ensureDir(pathOut)
    ensureDir(pathOutImgs)

    save_marking(numb_start, numb_end, step, templateImages, pathOutMarking)
    save_frames(numb_start, numb_end, step, pathOutImgs, templateImages)
    

 
