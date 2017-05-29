# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import sys
import subprocess
import re

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input', required=True, type=str, nargs='?', help='input hdf5 file with players')
parser.add_argument('-o', '--output', required=True, type=str, nargs='?', help='file to save result hdf5 file')
args = parser.parse_args()

in_file = args.input
out_file = args.output
tmp_out_dir = re.sub(r'.h5$', '', out_file) + '_tmp'
if not os.path.exists(tmp_out_dir):
    os.makedirs(tmp_out_dir)

out_image_file = os.path.join(tmp_out_dir, 'image.h5')
out_seg_file = os.path.join(tmp_out_dir, 'seg.h5')
out_depth_file = os.path.join(tmp_out_dir, 'depth.h5')

shutil.copy(in_file, out_image_file)

matlab2args_command = '{}(\'{}\',\'{}\');'

proc_depth = subprocess.Popen(['matlab', '-nodisplay',
                               '-nodisplay', ' -nosplash',
                               '-nodesktop', '-r',
                               matlab2args_command.format('estimateDepths', out_image_file, out_depth_file)],
                               stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL)
proc_seg = subprocess.Popen(['matlab', '-nodisplay',
                              '-nodisplay', ' -nosplash',
                              '-nodesktop', '-r',
                               matlab2args_command.format('estimateSegments', out_image_file, out_seg_file)],
                               stdin=subprocess.DEVNULL)

err = proc_depth.wait()
if err != 0:
    sys.stderr.write("ERROR!\n")
    exit(-1)

err = proc_seg.wait()
if err != 0:
    sys.stderr.write("ERROR!\n")
    exit(-1)

err = subprocess.run(['python', 'merge.py',
                      '-d', tmp_out_dir,
                      '-o', out_file])

if err.returncode != 0:
    sys.stderr.write("ERROR!\n")
    exit(-1)

shutil.rmtree(tmp_out_dir)

print("Database creation completed!")



