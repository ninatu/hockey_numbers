from synth_utils import NumberRender
from synth_text.common import colorize, Color

import argparse
import h5py
import traceback
import matplotlib.pyplot as plt
import tqdm


INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
DATA_PATH = 'synth_text/data'


def add_res_to_db(imgname,res,db):
    for i in range(len(res)):
        dname = "%s_%d"%(imgname, i)
        db['data'].create_dataset(dname,data=res[i]['img'])            
        db['data'][dname].attrs['txt'] = res[i]['txt']
        db['data'][dname].attrs['x'] = res[i]['x']
        db['data'][dname].attrs['y'] = res[i]['y']
        db['data'][dname].attrs['w'] = res[i]['w']
        db['data'][dname].attrs['h'] = res[i]['h']


def main():
    parser = argparse.ArgumentParser(description='Genereate Synthetic Player-Number Images')
    parser.add_argument('-i', '--inputDB', required=True, type=str, nargs='?', help='input database filename')
    parser.add_argument('-o', '--outputDB', required=True, type=str, nargs='?', help='output database filename')
    parser.add_argument('-n', '--count', default=INSTANCE_PER_IMAGE, type=int, help='instance per image')
    parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
    args = parser.parse_args()
    
    db_in = args.inputDB
    db_out = args.outputDB
    instance_per_instance = args.count
    viz = args.viz

    db = h5py.File(db_in,'r')

    out_db = h5py.File(db_out,'w')
    out_db.create_group('/data')
    
    imnames = sorted(db['image'].keys())
    start_idx,end_idx = 0, len(imnames)

    renderer = NumberRender(DATA_PATH)
    for i in tqdm.tqdm(range(start_idx,end_idx)):
        imname = imnames[i]

        img = db['image'][imname].value
        mask = db['mask'][imname].value
        all_res = []

        for i in range(instance_per_instance):
            try:
                res = renderer.render_text(img, mask, post_process=False)
                if res is not None:
                    all_res.append(res)
                    if viz:
                        plt.imshow(res['img']), plt.axis('off')
                        plt.show()
            except:
                traceback.print_exc()
                print(colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
                continue

        add_res_to_db(imname, all_res, out_db)

    db.close()
    out_db.close()


if __name__=='__main__':
    main()
