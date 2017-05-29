# -*- coding: utf-8 -*-

import cv2
import numpy as np


def retouch(img, mask):
    new_img = np.zeros(img.shape)
    counts = np.zeros(img.shape)

    h, w, d = img.shape
    # находим строки, которые покрывают маску
    rows_have_pixels = np.where(mask.sum(axis=1))[0]
    for row in rows_have_pixels:
        #находим столбы для данной строки, которые покрывают маску
        cols_have_pixels = np.where(mask[row])[0]
        min_col = min(cols_have_pixels)
        max_col = max(cols_have_pixels)
        l_color = img[row, max(0, min_col - 1)]
        r_color = img[row, min(w - 1, max_col + 1)]
        n = max_col - min_col + 2
        step = (r_color.astype(np.float) - l_color.astype(np.float)) / n
        new_color = [l_color + i * step for i in range(1, n)]
        new_color = np.array(new_color, dtype=np.uint8)
        new_img[row, min_col:max_col + 1] += new_color
        counts[row, min_col:max_col + 1] += np.ones(new_color.shape)
        
    # находим столбцы, которые покрывают маску
    cols_have_pixels = np.where(mask.sum(axis=0))[0]
    for col in cols_have_pixels:
        rows_have_pixels = np.where(mask[:, col])[0]
        min_row = min(rows_have_pixels)
        max_row = max(rows_have_pixels)
        t_color = img[max(0, min_row - 1), col]
        b_color = img[min(h - 1, max_row + 1), col]
        n = max_row - min_row + 2
        step = (b_color.astype(np.float) - t_color.astype(np.float)) / n
        new_color = [t_color + i * step for i in range(1, n)]
        new_color = np.array(new_color, dtype=np.uint8)
        new_img[min_row:max_row + 1, col] += new_color
        counts[min_row:max_row + 1, col] += np.ones(new_color.shape)

    out_img = img.copy()
    out_img[np.where(new_img)] = new_img[np.where(new_img)] / counts[np.where(new_img)]
    return out_img


def color_dilate_mask(img, mask, color_coeff):
    mask_pixels = img[np.where(mask)]
    centroid = np.average(mask_pixels, axis=0)
    n_centroid = centroid.shape[0]
    
    dist_to_centroid = np.linalg.norm(img - centroid.reshape((1, 1, n_centroid)), axis=2)
    mean_dist_in_mask = np.mean(np.linalg.norm(mask_pixels - centroid.reshape((1, n_centroid)), axis=1))
    color_dilate_mask = np.array(dist_to_centroid < (mean_dist_in_mask * color_coeff), dtype=np.uint8)
    return color_dilate_mask


def dist_dilate_mask(mask, dist_step):
    return cv2.dilate(mask, kernel=np.ones((dist_step, dist_step)))


def color_and_dist_dilate_mask(img, mask, color_coeff, dist_step):
    c_dilate = color_dilate_mask(img, mask, color_coeff)
    d_dilate = dist_dilate_mask(mask, dist_step)
    new_mask = np.array(c_dilate * d_dilate > 0, dtype=np.uint8)
    return new_mask
