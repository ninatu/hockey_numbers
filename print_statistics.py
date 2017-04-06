#!/usr/bin/python3

from libs.marking import Marking, MarkingStatictics
import numpy as np
import os.path as osp


markingFile = '/media/nina/Seagate Backup Plus Drive/hockey/marking/marking/sparse_frames/part3600_end13600_step100/marking.json'
outPath = osp.join(osp.dirname(markingFile), 'statistics.txt')

with open(outPath, 'w') as outFile:
    statictics = MarkingStatictics()
    statictics.addJsonMarking(markingFile)
    statictics.print(outFile)
    outFile.close()


