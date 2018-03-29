# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'

    ### Params:
        * `height`: Height of the feature map
        * `width`: Width of the feature map
        * `feat_stride`: feature stride
        * `anchor_scales`: default is (8, 16, 32) which means (8^2, 16^2, 32^2)
        * `anchor_ratios`: default is (0.5, 1, 2)

    ### Returns:
        * `anchors`: a float 2D array with shape of [K x A, 4], where K = w x h, 
                     A = len(anchor_ratio) x len(anchor_scale)
        * `length`: length of the all anchors, value = (w x h) x (3 x 3)
    """
    anchors = generate_anchors(ratios=np.array(
        anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]    # A = 9
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # ravel(): Return a flattened array.
    # shift length of [x1, y1, x2, y2]
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0] # w * h
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])     # K*A = (w*h) * (3*3)

    return anchors, length
