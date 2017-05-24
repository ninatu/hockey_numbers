import argparse
import os
import random
import sys
import subprocess
import re

DEFAULT_MAX_COUNT = 1e2

parser = argparse.ArgumentParser(description='')
parser.add_argument('--inpaint', action='store_true', help='get images with number and inpaint number')
parser.add_argument('-m', '--markingFile', required=True, type=str, nargs='?', help='file with marking of players')
parser.add_argument('-o', '--outputFile', required=True, type=str, nargs='?', help='file to save result hdf5 file')
parser.add_argument('-n', '--maxCount', required=True, type=int, nargs='?', help='max count of saving croped players')

args = parser.parse_args()


outFile = args.outputFile
tmpOutDir = re.sub(r'.h5$', '',  outFile) + '_tmp'
if not os.path.exists(tmpOutDir):
    os.makedirs(tmpOutDir)

outImageFile = os.path.join(tmpOutDir, 'image.h5')
outSegFile = os.path.join(tmpOutDir, 'seg.h5')
outDepthFile = os.path.join(tmpOutDir, 'depth.h5')

if args.inpaint:
    err = subprocess.run(['python', 'savePlayersToHDF5.py', 
                '-m', args.markingFile, 
                '-o', outImageFile, 
                '-n', str(args.maxCount), 
                '--inpaint'])
    if err.returncode != 0:
        sys.stderr.write("ERROR!\n")
        exit(-1)
else:
    err = subprocess.run(['python', 'savePlayersToHDF5.py', 
                '-m', args.markingFile, 
                '-o', outImageFile, 
                '-n', str(args.maxCount)])
    if err.returncode != 0:
        sys.stderr.write("ERROR!\n")
        exit(-1)

matlab2argsCommand = '{}(\'{}\',\'{}\');'
procDepth = subprocess.Popen(['matlab', '-nodisplay', 
                              '-nodisplay', ' -nosplash',
                              '-nodesktop', '-r',
                              matlab2argsCommand.format('estimateDepths', outImageFile, outDepthFile)],
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL)
procSeg = subprocess.Popen(['matlab', '-nodisplay', 
                              '-nodisplay', ' -nosplash',
                              '-nodesktop', '-r',
                              matlab2argsCommand.format('estimateSegments', outImageFile, outSegFile)],
                              stdin=subprocess.DEVNULL)

err = procDepth.wait()
if err != 0:
    sys.stderr.write("ERROR!\n")
    exit(-1)

err = procSeg.wait()
if err != 0:
    sys.stderr.write("ERROR!\n")
    exit(-1)


err = subprocess.run(['python', 'merge.py',
                '-d', tmpOutDir,
                '-o', outFile])
if err.returncode != 0:
    sys.stderr.write("ERROR!\n")
    exit(-1)

print("Database creation completed!")



