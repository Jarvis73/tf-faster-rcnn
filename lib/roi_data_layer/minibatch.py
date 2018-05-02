# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import numpy.random as npr
import cv2
import sys
sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from datasets.Liver_Kits import METType, abdominal_mask, raw_reader

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    if not cfg.MED_IMG:
        im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    else:
        im_blob, abdo_mask = _get_medical_image_blob(roidb)
        im_scales = [1] # compatible with original version

    blobs = {"data": im_blob, "abdo_mask": abdo_mask}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(
            roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
        dtype=np.float32)

    return blobs


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _get_medical_image_blob(roidb):
    """ Builds an input blob from the medical image in the roidb
    """
    num_images = len(roidb)
    processed_ims = []
    pre_ims = []
    post_ims = []
    abdo_masks = []
    for i in range(num_images):
        im = raw_reader(roidb[i]["image"], cfg.MET_TYPE, [roidb[i]["height"], roidb[i]["width"]])
        if roidb[i]['flipped']:
            im = im[:, ::-1]
        processed_ims.append(im)

        mask = abdominal_mask(im.copy())
        abdo_masks.append(mask)
        
        if cfg.THREE_SLICES:
            # get pre-image
            basename = osp.basename(roidb[i]["image"])
            names = basename[:-4].split("_")
            slice_num = int(names[-1])
            if slice_num == 0:
                pre_im = im
            else:
                slice_num -= 1
                names[-1] = str(slice_num)
                basename = "_".join(names) + ".raw"
                pre_path = osp.join(osp.dirname(roidb[i]["image"]), basename)
                pre_im = raw_reader(pre_path, cfg.MET_TYPE, [roidb[i]["height"], roidb[i]["width"]])
                if roidb[i]['flipped']:
                    pre_im = pre_im[:, ::-1]
            pre_ims.append(pre_im)

            # get post-image
            basename = osp.basename(roidb[i]["image"])
            names = basename[:-4].split("_")
            names[-1] = str(int(names[-1]) + 1)
            basename = "_".join(names) + ".raw"
            post_path = osp.join(osp.dirname(roidb[i]["image"]), basename)
            try:
                post_im = raw_reader(post_path, cfg.MET_TYPE, [roidb[i]["height"], roidb[i]["width"]])
                if roidb[i]['flipped']:
                    post_im = post_im[:, ::-1]
            except FileNotFoundError:
                post_im = im
            post_ims.append(post_im)

    num_images = len(processed_ims)
    blob = np.zeros((num_images, cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE, 3), dtype=np.float32)
    abdo_mask = np.zeros((num_images, cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE), dtype=np.bool)
    if cfg.THREE_SLICES:
        for i in range(num_images):
            blob[i,:,:,0] = pre_ims[i]
            blob[i,:,:,1] = processed_ims[i]
            blob[i,:,:,2] = post_ims[i]
            abdo_mask[i,:,:] = abdo_masks[i]
    else:
        for i in range(num_images):
            blob[i,:,:,0] = processed_ims[i]
            blob[i,:,:,1] = processed_ims[i]
            blob[i,:,:,2] = processed_ims[i]
            abdo_mask[i,:,:] = abdo_masks[i]

    if cfg.USE_WIDTH_LEVEL:
        win, wind2, lev = cfg.WIDTH, cfg.WIDTH / 2, cfg.LEVEL
        blob = (np.clip(blob, lev - wind2, lev + wind2) - (lev - wind2)) / 2**16 * win
    else:
        blob /= cfg.MED_IMG_UPPER
        blob = np.clip(blob, -1., 1.)

    return blob, abdo_mask

if __name__ == "__main__":
    roidb = [{"image": "C:/DataSet/LiverQL/liver_2017_train/liver/Q001_o_57.raw",
              "height": 512,
              "width": 512,
              "flipped": False}]
    blob, abdo_mask = _get_medical_image_blob(roidb)