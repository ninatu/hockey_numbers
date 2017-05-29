# -*- coding: utf-8 -*-

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, exists, dirname
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from segmentation import get_masks_of_number, get_masks_of_segments, fill_black_pixels, color_segmentation
from feature_extraction import extract_features
from constants import PATH_TO_CLASSIFIER


#image_dir = '/home/nina/Documents/hockey_tracking/number_recognition/number_recognition/data/seg4_lab'
#clf_path = join(dirname(__file__), 'data/maskClf3.plk')


def file_paths(directory):
    files = listdir(directory)
    files = map(lambda f: join(directory, f), files)
    return [f for f in files if isfile(f)]


def array_is_contained(matr, list_of_matr):
    return any(np.array_equal(matr, x) for x in list_of_matr)


def eq_mask(mask1, mask2, thresh=5):
    return sum(sum(mask1.astype(np.bool) != mask2.astype(np.bool))) < thresh


def get_unique_masks(in_list):
    uniq = []
    for x in in_list:
        have = False
        for y in uniq:
            if eq_mask(x, y):
                have = True
                break
        if not have:
            uniq.append(x)
    return uniq


def create_classifier(image_dir, out_path):
    X_true, all_X_false = [], []
    img_path = file_paths(image_dir)
    for fname in img_path:
        seg = cv2.imread(fname)
        masks_true = get_masks_of_number(seg)
        masks_true = get_unique_masks(masks_true)
        X_true.extend(masks_true)
        masks_all = get_masks_of_segments(fill_black_pixels(seg))
        masks_all_false = filter(lambda x: not array_is_contained(x, masks_true), masks_all)
        all_X_false.extend(masks_all_false)

    X_true = extract_features(X_true)
    all_X_false = extract_features(all_X_false)

    X_train = np.concatenate((X_true, all_X_false), axis=0)
    y_train = np.concatenate((np.ones(X_true.shape[0]), np.zeros(all_X_false.shape[0])))
    n_true = sum(y_train)
    n_false = sum(y_train == 0)
    estimator = RandomForestClassifier(100, class_weight={1:n_true / (n_true + n_false), 0: n_false / (n_true + n_false)})

    estimator.fit(X_train, y_train)
    joblib.dump(estimator, out_path)


def get_classifier():
    clf = joblib.load(PATH_TO_CLASSIFIER)
    return clf





"""
def findMaskOfNumber(img, imMask, clf):
    img = bgr2lab(img)
    colorSeg = colorSegByMask(img, mask=imMask, nColors=4)
    masksOfSeg = get_masks_of_segments(colorSeg)
    #masksOfSeg = [morphologyClosing(x) for x in masksOfSeg]
    X = extract_features(masksOfSeg)
    y = clf.predict_proba(X)[:,1]
    arg_y = np.argsort(y)[::-1]
    maskOfNumber = []
    #for i in  arg_y[:3]:
    #    maskOfNumber.append((masksOfSeg[i],y[i]))
    
    for i, mask in enumerate(masksOfSeg):
        if y[i] >= 0.5:
            maskOfNumber.append((mask, y[i]))
    
    return maskOfNumber
"""