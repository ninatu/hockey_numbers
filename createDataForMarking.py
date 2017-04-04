#!/usr/bin/python3

from libs.marking import BaseMarkingCreator
import os.path as osp
import os
import argparse
import sys
import cv2

parser = argparse.ArgumentParser(description='Prepare data for marking')
parser.add_argument('-f', '--first', required=True, type=int, nargs='?', help='number of start frame')
parser.add_argument('-c', '--count', required=True, type=int, nargs='?', help='count of frames')
parser.add_argument('-s', '--step', required=True, type=int, nargs='?', help='step in frames')

#saving frames
def save_frames(number_start, number_end, step, out_folder, out_template):
    video_dir = '/media/nina/Seagate Backup Plus Drive/hockey/processed_video/'
    video_template = "cska_akbars_cam_3_{:d}_{:d}.avi"
    number_frames = set(range(numb_start, numb_end, step))

    cur_start = int(numb_start / 1200) * 1200
    while(True):
        in_video = video_dir + video_template.format(cur_start, cur_start + 1199)
        reader = cv2.VideoCapture(in_video)

        for i_frame in range(cur_start, cur_start+1200):
            ret, frame = reader.read()
            if not ret:
                break
            if i_frame in number_frames:
                cv2.imwrite(osp.join(out_folder, out_template.format(i_frame)), frame)
        reader.release()
        cur_start += 1200
        if cur_start >= numb_end:
            break

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

    save_marking(numb_start, numb_end, step, templateImages, pathOutMarking)
    save_frames(numb_start, numb_end, step, pathOutImgs, templateImages)
    

 
