import argparse
import h5py
import os.path as osp
import numpy as np
from segmentation import segmentFromContourMap

DEFAULT_THRESHOLD = 0.11

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--inputDirectory', required=True, type=str, nargs='?', help='input directory with image.h5, seg.h5, depth.h5 files')
parser.add_argument('-o', '--outputFile', required=True, type=str, nargs='?', help='file to save result')
parser.add_argument('-t', '--threshold', default=DEFAULT_THRESHOLD, type=float, nargs='?', help='threshold for contour in segmentation, default = '+str(DEFAULT_THRESHOLD))

args = parser.parse_args()
inDir = args.inputDirectory
outPath = args.outputFile
segThresh = args.threshold

try :
    imgDB = h5py.File(osp.join(inDir, 'image.h5'), 'r')
    segDB = h5py.File(osp.join(inDir, 'seg.h5'), 'r')
    depthDB = h5py.File(osp.join(inDir, 'depth.h5'), 'r')
    
    imgs = imgDB['image']
    segs= segDB['seg']
    depths = depthDB['depth']
    
    imgNames = sorted(imgDB['image'].keys())

    outDB = h5py.File(outPath, 'w')
    imageGr = outDB.create_group('image')
    segGr = outDB.create_group('seg')
    depthGr = outDB.create_group('depth')    
    
    for imgName in imgNames:
        img = imgs[imgName].value
        depth = depths[imgName].value
        seg = segs[imgName].value
        seg = np.array(seg > segThresh, dtype=np.uint8)
        seg, label, area  = segmentFromContourMap(seg)
        seg = np.array(seg, dtype=np.uint16)

        _imset = imageGr.create_dataset(imgName, img.shape, '|u1', img)
        _depthset = depthGr.create_dataset(imgName, depth.shape, '<f4', depth)
        _segset = segGr.create_dataset(imgName, seg.shape, "<u2", seg)
        _segset.attrs['label'] = label
        _segset.attrs['area'] = area        
        #saving attrs
        imgAttrs = imgs[imgName].attrs
        for key in imgAttrs.keys():
            _imset.attrs[key] = imgAttrs[key]

    imgDB.close()
    segDB.close()
    depthDB.close()
    outDB.close()
except Exception as e:
    print(str(e))

  



