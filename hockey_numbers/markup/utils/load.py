#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Package for load masks and frames"""

import scipy.misc
import os.path as osp
import numpy as np

from hockey_numbers.markup.constants import FRAME_DIR, TEMPLATE_FRAME, MASK_DIR, TEMPLATE_MASK, TEMPLATE_IMAGE


def load_frame(numb):
    """Return None if frame is not saved"""

    frame_name = osp.join(FRAME_DIR, TEMPLATE_FRAME.format(numb))
    frame = scipy.misc.imread(frame_name)
    return frame

def load_mask(numb):
    """Return None if mask is not saved"""

    mask_name = osp.join(MASK_DIR, TEMPLATE_MASK.format(numb))
    mask = scipy.misc.imread(mask_name)
    return mask