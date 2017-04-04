#!/usr/bin/python3

from libs.marking import BaseMarkingCreator
import os
import argparse
import sys
import subprocess

parser = argparse.ArgumentParser(description='Prepare data for marking')
parser.add_argument('-s', '--start', required=True, type=int, nargs='?', help='number of start frame')
parser.add_argument('-c', '--count', required=True, type=int, nargs='?', help='count of frames')

args = parser.parse_args(sys.argv[1:])

numb_start = args.start
count = args.count

masksDir = '/media/nina/Seagate Backup Plus Drive/hockey/masks/'
videoDir = '/media/nina/Seagate Backup Plus Drive/hockey/processed_video/'

templateImages = 'cska_akbars_{:d}.jpg'
templateMasks = 'mask{:d}.png'
pathOut = '/media/nina/Seagate Backup Plus Drive/hockey/marking/sourceForMarking/part%d_%d'%(numb_start, numb_start + count)
pathOutImgs = os.path.join(pathOut, 'imgs')
pathOutMarking = os.path.join(pathOut, 'marking.json')

def ensureDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
ensureDir(pathOut)
ensureDir(pathOutImgs)

subprocess.run(['python', os.path.join(os.path.dirname(__file__), 'saveFrames.py'), '-s',  str(numb_start), 
		'-c', str(count), '-o', pathOutImgs])

marking = BaseMarkingCreator(masksDir, templateMasks, templateImages)
marking.addFrames(numb_start, numb_start + count, step=4)
marking.filterBySize(70, 190, 40, 120)
marking.filterByField(0, 820, 0, 5700)
marking.saveAsJson(open(pathOutMarking, 'w'))

