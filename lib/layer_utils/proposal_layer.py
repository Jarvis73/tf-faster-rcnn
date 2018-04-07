# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors, abdo_mask, only_rpn=False):
    """
    A simplified version compared to fast/er RCNN. For details please see the technical report
    
    Params
    ---
    `rpn_cls_prob`: class scores of all the anchors, the shape is [bs, h, w, 2*9]  
    `rpn_bbox_pred`: bbox parameters of all the anchors, the shape is [bs, h, w, 4*9]  
    `im_info`: a tensor array with three number, [image height, image width, image resize ratio
    for medical images, image resize ratio equals to 1.0 always, because they are the same 
    size 512 x 512  
    `cfg_key`: mode, TRAIN or TEST  
    `_feat_stride`: stride from the input images to feature maps in head, for example it equals 
    to 16 for VGG16  
    `anchors`: all the anchors computed in `anchor_component()`  
    `num_anchors`: number of anchors at single point  

    Returns
    ---
    `blob`: output or rpn, rois used for Fast R-CNN
    `scores`: scores of blob
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    if not only_rpn:
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N       # 12000 for train, 6000 for test
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N     # 2000  for train, 300  for test
    else:
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N_ONLY_RPN 
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N_ONLY_RPN
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH            # 0.7

    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

    if cfg.ONLY_INSIDE_ABDOMEN:
        anchor_centers = ((proposals[:, 0:2] + proposals[:, 2:4]) / 2).astype(np.int32)
        shape = abdo_mask.shape[1:3]
        cond1 = np.logical_and(anchor_centers[:,0] >= 0, anchor_centers[:,0] < shape[1])
        cond2 = np.logical_and(anchor_centers[:,1] >= 0, anchor_centers[:,1] < shape[0])
        inds_inside_bounding = np.where(np.logical_and(cond1, cond2))[0]
        inds_inside_abdomen = np.where(abdo_mask[0][anchor_centers[inds_inside_bounding, 1], \
            anchor_centers[inds_inside_bounding, 0]])[0]
        inds_inside = inds_inside_bounding[inds_inside_abdomen]
        proposals = proposals[inds_inside, :]
        scores = scores[inds_inside, :]
    proposals = clip_boxes(proposals, im_info[:2])

    # Pick the top N region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    # return keeped indices
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
