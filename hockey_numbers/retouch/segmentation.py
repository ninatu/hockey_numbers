# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sklearn.cluster import KMeans


def color_segmentation(rgb_image, n_colors, mask):
    h, w, d = rgb_image.shape
    train_pixels = rgb_image[np.where(mask)]
    pixels = rgb_image.reshape((h * w, 3))

    clt = KMeans(n_clusters=n_colors)
    clt.fit(train_pixels)
    pix_labels = clt.predict(pixels)
    color_labels = clt.cluster_centers_
    segmented_pixels = np.empty(pixels.shape, dtype=np.uint8)

    for i, color in enumerate(color_labels):
        segmented_pixels[np.where(pix_labels == i)] = color

    seg_image = segmented_pixels.reshape((h, w, d))
    return seg_image


def flatten_color_segment(img):
    h, w, d = img.shape
    assert d == 3
    return img[:,:,0] + img[:,:,1] * 256 + img[:,:,2] * 256 * 256


def number_connect_components(mapp, connectivity=4):
    mapshape = mapp.shape
    assert len(mapshape) == 2
    uniq_marks = np.unique(mapp)
    segment_marking = np.zeros(mapp.shape, dtype=np.int64)
    cur_n = 0
    for i in uniq_marks:
        cur_mask_segments = np.array(mapp == i, dtype=np.uint8)
        n_seg, i_seg_marking = cv2.connectedComponents(cur_mask_segments, connectivity=connectivity)
        #добавляем новые пронумерованные сегменты к результату,
        #для этого нужно увеличить номер каждого сегмента на кол-во уже занятых
        segment_marking += (i_seg_marking + (cur_mask_segments * cur_n))
        cur_n += (n_seg - 1)
    return segment_marking


def get_masks_of_segments(in_img):
    f_img = flatten_color_segment(in_img)
    seg = number_connect_components(f_img)
    uniq_marks = np.unique(seg)
    masks = []
    for mark in uniq_marks:
        if mark == 0:
            continue
        masks.append(np.array(seg == mark, dtype=np.uint8))
    return masks


def get_masks_of_number(in_img):
    """
    На каждом цветовом сегменте, принадлежащему к цифре, была поставлена черная точка.
    Данная функция возвращает маски сегментов, принажлежаших цифрам.
    """
    color_seg = flatten_color_segment(in_img)
    seg = number_connect_components(color_seg)

    black_pixel_mask = np.all(in_img == 0, axis=2)
    black_pixel_coords = np.argwhere(black_pixel_mask)
    masks = []
    for (r, c) in black_pixel_coords:
        mark = seg[r - 1, c]
        seg[r, c] = mark
        masks.append(np.array(seg == mark, np.uint8))
    return masks


def fill_black_pixels(in_img):
    cur_img = in_img.copy()
    black_pixel_mask = np.all(in_img == 0, axis=2)
    black_pixel_coords = np.argwhere(black_pixel_mask)
    for (r, c) in black_pixel_coords:
        color = cur_img[r - 1, c]
        cur_img[r, c, :] = color
    return cur_img