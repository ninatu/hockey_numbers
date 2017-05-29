# Author: Ankush Gupta
# Date: 2015
# Modified by Nina Tuluptseva

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }

"""

import numpy as np
import h5py
import sys, traceback
import os.path as osp
from SynthText.synthgen import *
from SynthText.common import *
from constants import SYNTH_TEXT_DATA_DIR, DEFAULT_INSTANCE_PER_IMAGE, SECS_PER_IMG


def get_data(inputDB):
    if not osp.exists(inputDB):
        print colorize(Color.RED,'Data not found.',bold=True)
        sys.stdout.flush()
        sys.exit(-1)
    return h5py.File(inputDB,'r')


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    db['data'][dname].attrs['txt'] = res[i]['txt']


def main(args):
  # open databases:
  print colorize(Color.BLUE,'getting data..',bold=True)
  db = get_data(args.input)
  print colorize(Color.BLUE,'\t-> done',bold=True)

  # open the output h5 file:
  out_db = h5py.File(args.output,'w')
  out_db.create_group('/data')
  print colorize(Color.GREEN,'Storing the output in: '+ args.output, bold=True)

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  start_idx,end_idx = 0, N

  RV3 = RendererV3(SYNTH_TEXT_DATA_DIR, max_time=SECS_PER_IMG)
  for i in xrange(start_idx,end_idx):
    imname = imnames[i]
    try:
      # get the image:
      img = Image.fromarray(db['image'][imname][:])
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth = db['depth'][imname][:].T
      depth = depth[:,:,1]
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz, Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

      print colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True)
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=args.count,viz=args.viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(imname,res,out_db)
      # visualize the output:
      if args.viz:
        if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
      continue
  db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('-i', '--input', required=True, type=str, nargs='?', help='input database filename')
  parser.add_argument('-o', '--output', required=True, type=str, nargs='?', help='output filename')
  parser.add_argument('-n', '--count', default=DEFAULT_INSTANCE_PER_IMAGE, type=int, help='instance per image')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args)
