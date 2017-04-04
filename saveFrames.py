import argparse
import sys
import cv2
import os.path as osp

parser = argparse.ArgumentParser(description='Saving frames frim video')
parser.add_argument('-s', '--start', type=int, nargs='?', help='number of start frame')
parser.add_argument('-c', '--count', type=int, nargs='?', help='count of frames')
parser.add_argument('-o', '--outdir', type=str, nargs='?', help='dir to save frames')
args = parser.parse_args(sys.argv[1:])



numb_start = args.start
count = args.count
out_folder = args.outdir

out_template = "cska_akbars_{:d}.jpg"
video_template = "cska_akbars_cam_3_{:d}_{:d}.avi"
video_dir = "/media/nina/Seagate Backup Plus Drive/hockey/processed_video/"

in_video = video_dir + video_template.format(numb_start, numb_start+1199)
reader = cv2.VideoCapture(in_video)

for i_frame in range(numb_start, numb_start + count):
	ret, frame = reader.read()
	if not ret:
	    break
	cv2.imwrite(osp.join(out_folder, out_template.format(i_frame)), frame)
reader.release()

