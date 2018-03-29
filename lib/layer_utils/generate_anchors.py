# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.

    wrt: with respect to 关于
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    #                                 h : w
    # [[ -3.5   2.   18.5  13. ]    0.5 : 1
    #  [  0.    0.   15.   15. ]      1 : 1
    #  [  2.5  -3.   12.5  18. ]]     2 : 1
    
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    # h * scales, w * scales
    #                               ratio     scales      size     area 16*16* 
    # [[ -84.  -40.   99.   55.]    0.5:1       8      [ 96, 184]    8**2
    #  [-176.  -88.  191.  103.]    0.5:1      16      [192, 368]   16**2
    #  [-360. -184.  375.  199.]]   0.5:1      32      [384, 736]   32**2
    # [[ -56.  -56.   71.   71.]      1:1       8      [128, 128]    8**2
    #  [-120. -120.  135.  135.]      1:1      16      [256, 256]   16**2
    #  [-248. -248.  263.  263.]]     1:1      32      [512, 512]   32**2
    # [[ -36.  -80.   51.   95.]      2:1       8      [176,  88]    8**2
    #  [ -80. -168.   95.  183.]      2:1      16      [352, 176]   16**2
    #  [-168. -344.  183.  359.]]     2:1      32      [704, 352]   32**2

    return anchors


# width height centers
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    #     16,16, 7.5,   7.5
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


# ratio enumerate
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h    # 256 = 16*16
    size_ratios = size / ratios     # 256 / [0.5, 1, 2] = [512, 256, 128]
    ws = np.round(np.sqrt(size_ratios))     # [23., 16., 11.]
    hs = np.round(ws * ratios)              # [12., 16., 22.]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    # from IPython import embed

    # embed()
