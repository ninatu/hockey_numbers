from synth_text.text_utils import RenderFont
from synth_text.colorize3_poisson import Colorize

import random
import numpy as np
import skimage.transform
import scipy.misc
import cv2
from math import sin, cos, radians


def sin_transform(image, n=10, x=0):
    image = image.astype(np.float)
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(x, 2 * np.pi + x, src.shape[0])) * n

    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= 1.5 * n
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = skimage.transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out = skimage.transform.warp(image, tform)
    out = scipy.misc.imresize(out, image.shape)
    out = out.astype(np.uint8)

    return out


def get_orient_player(mask):
    cnt, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.concatenate(cnt, axis=0)
    ellipse = cv2.fitEllipse(cnt)
    center, size, angle = ellipse

    angle = (angle + 90) % 180 - 90
    return center, size, angle


def get_orient_text(mask, text):
    cnt, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.concatenate(cnt, axis=0)
    ellipse = cv2.fitEllipse(cnt)
    center, size, angle = ellipse

    angle = (angle + 90) % 180 - 90

    # stable orientation text
    if len(text) > 1 and size[0] < size[1] and abs(angle) > 45:
        angle = angle - 90 if angle > 0 else angle + 90
        size = (size[1], size[0])

    return center, size, angle


def crop(mask):
    cnt, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.concatenate(cnt, axis=0)
    x, y, w, h = cv2.boundingRect(cnt)
    return mask[y:y + h, x:x + w]


def add_pad(mask, pad=1.0):
    h, w = mask.shape
    pad_h, pad_w = int(pad * h), int(pad * w)
    pad_mask = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.uint8)
    pad_mask[pad_h:pad_h + h, pad_w:pad_w + w] = mask
    return pad_mask


def correct_orient(mask, angle, scale):
    h, w = mask.shape
    rot_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, scale)
    return cv2.warpAffine(mask, rot_matrix, (w, h))  # , (cols, rows))


def sample_image(img, max_pad=5, max_sigma=1.1, max_kernel=11):
    pad_x1 = random.randint(0, max_pad + 1)
    pad_x2 = random.randint(0, max_pad + 1)
    pad_y1 = random.randint(0, max_pad + 1)
    pad_y2 = random.randint(0, max_pad + 1)
    kernel = random.choice(range(1, max_kernel + 1, 2))
    sigma = max_sigma * random.random()

    h, w, c = img.shape
    img = img[pad_y1:h - pad_y2, pad_x1:w - pad_x2]
    if kernel != 1:
        img = cv2.GaussianBlur(img, (kernel, kernel), sigma)
    return img.astype(np.uint8), (pad_x1, pad_y1)

class NumberRender:
    def __init__(self, data_dir='synth_text/data'):
        self.text_renderer = RenderFont(data_dir)
        self.colorizer = Colorize(data_dir)

        # self._alpha_number = 0.6
        self._min_shift = 0.10
        self._max_shift = 0.25
        self._min_scale = 0.7
        self._max_scale = 1
        self._color_h = 3
        self._max_attempt = 15

    def sample_text(self):
        collision_mask = np.zeros((70, 70),  # (70, 70),
                                  dtype=np.uint8)  # text_renderer.max_font_h *3, text_renderer.max_font_h * 4 ), dtype=np.uint)
        font = self.text_renderer.font_state.sample()
        font = self.text_renderer.font_state.init_font(font)
        render_res = self.text_renderer.render_sample(font, collision_mask)
        if render_res:
            text_mask, loc, bb, text = render_res
            return text_mask, text
        else:
            return None

    def sample_transform(self, mask):
        mask = crop(mask)
        mask = add_pad(mask, pad=1.2)
        mask = sin_transform(mask, n=random.randint(10, 15), x=random.randint(1, 10))
        return mask

    def sample_shift(self):
        return self._min_shift + (self._max_shift - self._min_shift) * random.random()

    def sample_scale(self, text):
        scale = self._min_scale + (self._max_scale - self._min_scale) * random.random()
        if len(text) == 1:
            scale /= 1.2
        return scale

    def colorize(self, rgb, text_mask):
        return self.colorizer.color(rgb, [text_mask], np.array([self._color_h]))

    def correct_orient(self, img_text, text, mask_player):
        n_cntr, n_sz, n_ang = get_orient_text(img_text, text)
        p_cntr, p_sz, p_ang = get_orient_player(mask_player)

        img_text = crop(img_text)
        img_text = add_pad(img_text, pad=2)

        cor_angle = -(p_ang - n_ang)
        # cor_scale = p_sz[0] / n_sz[0] * self._scale_number
        img_text = correct_orient(img_text, cor_angle, 1)  # cor_scale)
        return img_text

    def check_mask(self, text, mask):
        text = np.array(text, dtype=np.bool)
        mask = np.array(mask, dtype=np.bool)
        text_in = np.logical_and(text, mask)
        return not (np.logical_xor(text, text_in).sum())

    def place_text(self, text_mask, text, img_player, mask_player):

        text_mask = crop(text_mask)
        p_cntr, p_sz, p_ang = get_orient_player(mask_player)
        n_cntr, n_sz, n_ang = get_orient_text(text_mask, text)

        for i in range(self._max_attempt):
            alpha = min(p_sz) / max(n_sz) * self.sample_scale(text)
            text_mask = add_pad(text_mask, pad=0.05)

            h, w = map(lambda x: int(x * alpha), text_mask.shape)
            sample_text = scipy.misc.imresize(text_mask, (h, w))

            alpha_shift = self.sample_shift()
            p_h = p_sz[1]
            pcntr_x, pcntr_y = (p_cntr[0], p_cntr[1])
            pcntr_x = pcntr_x - p_h * alpha_shift * sin(radians(-p_ang))
            pcntr_y = pcntr_y - p_h * alpha_shift * cos(radians(-p_ang))

            h, w = sample_text.shape
            tcntr_x, tcntr_y = (w // 2, h // 2)
            x = int(pcntr_x - tcntr_x)
            y = int(pcntr_y - tcntr_y)
            p_h, p_w = mask_player.shape

            if not (x > 0 and y > 0 and x + w < p_w and y + h < p_h):
                continue

            if self.check_mask(sample_text, mask_player[y:y + h, x:x + w]):
                res_img = img_player.copy()
                res_crop = self.colorize(res_img[y:y + h, x:x + w], sample_text)
                res_img[y:y + h, x:x + w] = res_crop
                return res_img, (x, y, w, h)

        return (None, (None, None, None, None))

    def render_text(self, img_player, mask_player, post_process=True):
        res = self.sample_text()
        if res is None:
            return None
        text_mask, text = res

        text_mask = self.sample_transform(text_mask)
        #if len(text) > 1:
        #    text_mask = self.correct_orient(text_mask, text, mask_player)

        res_img, (x, y, w, h) = self.place_text(text_mask, text, img_player, mask_player)
        if res_img is not None:
            if post_process:
                res_img, (d_x, d_y) = sample_image(res_img)
                x -= d_x
                y -= d_y
                
            res_data = {}
            res_data['img'] = res_img
            res_data['x'] = x
            res_data['y'] = y
            res_data['w'] = w
            res_data['h'] = h
            res_data['txt'] = text
            return res_data

        return None

