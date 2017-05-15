#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Package for load masks and frames"""

import scipy.misc
import os.path as osp

from constants import FRAME_DIR, TEMPLATE_FRAME, MASK_DIR, TEMPLATE_MASK


def load_frame(numb=None, name=None):
    """Return None if frame is not saved"""
    assert (numb is not None) or (name is not None)

    if numb:
        name = osp.join(FRAME_DIR, TEMPLATE_FRAME.format(numb))

    frame = scipy.misc.imread(name)
    return frame


def load_mask(numb=None, name=None):
    """Return None if mask is not saved"""
    assert (numb is not None) or (name is not None)

    if numb:
        name = osp.join(MASK_DIR, TEMPLATE_MASK.format(numb))

    mask = scipy.misc.imread(name)
    return mask
