import argparse
import os
import random
import sys
import subprocess
import re

DEFAULT_MAX_COUNT = int(1e9)

parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', '--outputFile', required=True, type=str, nargs='?', help='file to save result hdf5 file')
parser.add_argument('-n ', '--maxCount', default=DEFAULT_MAX_COUNT, type=int, nargs='?', help='max count of saving croped players')
args = parser.parse_args()


outFile = args.outputFile
tmpOutDir = re.sub(r'.h5$', '',  outFile) + '_tmp'
if not os.path.exists(tmpOutDir):
    os.makedirs(tmpOutDir)

outImageFile = os.path.join(tmpOutDir, 'image.h5')
outSegFile = os.path.join(tmpOutDir, 'seg.h5')
outDepthFile = os.path.join(tmpOutDir, 'depth.h5')

subprocess.run(['python', 'savePlsWithInpainting.py', 
                '-o', outImageFile, 
                '-n', str(args.maxCount)])

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

procDepth.wait()
procSeg.wait()

subprocess.run(['python', 'merge.py',
                '-d', tmpOutDir,
                '-o', outFile])
os.remove(outImageFile)
os.remove(outSegFile)
os.remove(outDepthFile)
os.rmdir(tmpOutDir)

print("Database creation completed!")



