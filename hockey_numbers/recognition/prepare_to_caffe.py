import numpy as np
import lmdb
import caffe
import h5py
import os.path as osp
import scipy.misc
import skimage
import os
import argparse
import tqdm
import sys


def writeToLMDB(env, readers, start_i, h, w, convertToGray=False):
    i = start_i
    txn = env.begin(write=True)
    for reader in readers:
        for img, number in reader.images():
            datum = caffe.proto.caffe_pb2.Datum()
            datum.height = h
            datum.width = w
            img = scipy.misc.imresize(img, (h, w))
            img = img.astype(np.uint8)

            if convertToGray:
                datum.channels = 1
                img = skimage.color.rgb2gray(img, cv2.COLOR_BGR2GRAY)
                img = img.reshape((1, h, w))
            else:
                datum.channels = 3
                res_img = np.empty((3, h, w), dtype=np.uint8)
                res_img[0, :, :] = img[:, :, 0]
                res_img[1, :, :] = img[:, :, 1]
                res_img[2, :, :] = img[:, :, 2]
                img = res_img

            datum.data = img.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(number)
            str_id = '{:07}'.format(i)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            i += 1
            if (i % 1000 == 0):
                txn.commit()
                env.sync()
                txn = env.begin(write=True)
    if (i % 1000 != 0):
        txn.commit()
        env.sync()

class SynthTextImageReader:

    def __init__(self, path):
        self._path = path

    def images(self):
        inDB = h5py.File(self._path)
        img_names = list(inDB['data'].keys())

        for img_name in tqdm.tqdm(img_names):
            img_data = inDB['data'][img_name]
            img = img_data.value
            number = int(img_data.attrs['txt'][0].decode('utf-8'))
            yield (img, number)

class DirReader:

    def __init__(self, path):
        self._path = path

    def images(self):

        subdirs = os.listdir(self._path)
        subdirs = filter(lambda x: osp.isdir(osp.join(self._path, x)), subdirs)

        for subdir in tqdm.tqdm(subdirs):
            number = int(subdir)

            subdir = osp.join(self._path, subdir)
            imgfiles = os.listdir(subdir)
            imgfiles = map(lambda x: osp.join(subdir, x), imgfiles)
            imgfiles = filter(lambda x: osp.isfile(x), imgfiles)

	    for imgfile in imgfiles:
	        if not osp.isfile(imgfile):
	            continue
                img = scipy.misc.imread(imgfile)
                yield (img, number)
		

def prepare(synthtext, dirs, outfile, h, w, append=False, to_gray=False):
    assert not osp.exists(outfile) and append == False

    readers = [SynthTextImageReader(infile) for infile in synthtext]
    readers.extend([DirReader(indir) for indir in dirs])

    channels = 3 if not to_gray else 1
    map_size = 10 ** 9 * h * w * channels

    env = lmdb.open(outfile, map_size=map_size)
    start_i = 0 if not append else env.stat()["entries"]

    writeToLMDB(env=env, readers=readers, start_i=start_i, h=h, w=w, convertToGray=to_gray)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--synthtext', nargs='*', type=str, default=[], help="synth text hdf5 output")
    parser.add_argument('--dirnames', nargs='*', type=str, default=[], help='dirs with images')
    parser.add_argument('-t', '--height', nargs='?', type=int, required=True, help='height of result images')
    parser.add_argument('-w', '--width', nargs='?', type=int, required=True, help='width of result images')
    parser.add_argument('--append', action='store_true', help='true is append data to file')
    parser.add_argument('--togray', action='store_true', help='convert to gray')
    parser.add_argument('-o', '--outfile', nargs='?', type=str, help='file to save result')

    args = parser.parse_args()

    prepare(synthtext=args.synthtext, dirs=args.dirnames,
            outfile=args.outfile,
            h=args.height, w=args.width,
            append=args.append, to_gray=args.togray)

if __name__ == '__main__':
    main()
