#!/usr/bin/python3

from libs.marking import Marking, MarkingStatictics
import numpy as np
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('marking_file', nargs='?', help='input file with marking')
args = parser.parse_args()

markingFile = args.marking_file
outPath = osp.join(osp.dirname(markingFile), 'statistics.txt')

with open(outPath, 'w') as outFile:
    statictics = MarkingStatictics()
    statictics.addJsonMarking(markingFile)
    statictics.print(outFile)
    outFile.close()


