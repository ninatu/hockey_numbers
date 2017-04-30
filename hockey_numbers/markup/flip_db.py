import h5py
from utils.image_process import md5_hash
import cv2
import tqdm


def flip_db(in_db_name, out_db_name):
    in_db = h5py.File(in_db_name)

    out_db = h5py.File(out_db_name, 'w')
    out_db.create_group('image')
    out_db.create_group('mask')

    for img_name in tqdm.tqdm(in_db['image'].keys()):
        img = in_db['image'][img_name].value
        mask = in_db['mask'][img_name].value

        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        img_name = '{}.png'.format(md5_hash(img))
        out_db['image'].create_dataset(img_name, img.shape, '|u1', img)
        out_db['mask'].create_dataset(img_name, mask.shape, '|u1', mask)


#in_db = '/media/nina/Seagate Backup Plus Drive/hockey/blobs_data/part_40000_110000_not_number.h5'
#out_db = '/media/nina/Seagate Backup Plus Drive/hockey/blobs_data/part_40000_110000_not_number_flip.h5'
#flip_db(in_db, out_db)

