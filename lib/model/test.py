# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import math

from datasets.liverQL import liverQL
from datasets.Liver_Kits import mhd_reader, abdominal_mask
from datasets.imdb import imdb as IMDB

from utils.timer import Timer
from utils.blob import im_list_to_blob
from utils.average_precision import DetectionMAP

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_med_image_blob(im):
    im_shape = im.shape
    blob = np.zeros((1, im_shape[0], im_shape[1], 3), dtype=np.float32)
    for i in range(3):
        blob[0,:,:,i] = im
    if cfg.USE_WIDTH_LEVEL:
        win, wind2, lev = cfg.WIDTH, cfg.WIDTH / 2, cfg.LEVEL
        blob = (np.clip(blob, lev - wind2, lev + wind2) - (lev - wind2)) / 2**16 * win
    else:
        blob /= cfg.MED_IMG_UPPER
        blob = np.clip(blob, -1., 1.)
    
    return blob, np.ones((1), dtype=np.float32)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    if not cfg.MED_IMG:
        blobs['data'], im_scale_factors = _get_image_blob(im)
    else:
        blobs['data'], im_scale_factors = _get_med_image_blob(im)
        blobs['abdo_mask'] = np.reshape(abdominal_mask(im.copy()), (-1, 512, 512))
        blobs['im_info'] = np.array([blobs['data'].shape[1], blobs['data'].shape[2], 
                                    im_scale_factors[0]], dtype=np.float32)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    if cfg.ONLY_RPN:
        rois, scores = net.test_rpn_image(sess, blobs)
        pred_boxes = np.zeros((rois.shape[0], 8), dtype=np.float32)
        pred_boxes[:, 4:8] = rois[:, 1:5]
        scores = np.reshape(scores, [scores.shape[0], -1])
    else:
        _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
        boxes = rois[:, 1:5] / im_scales[0]
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = _clip_boxes(pred_boxes, im.shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(sess, net, 
             imdbs, 
             weights_filename, 
             max_per_image=100, 
             thresh_nms=0.5,
             thresh_map=0.5):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = [im.num_images for im in imdbs]
    num_images = np.sum(num_images)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(imdbs[0].num_classes)]

    imdb = IMDB("+".join([im.name for im in imdbs])) 
    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    accu = 0
    mAP = DetectionMAP(imdbs[0].num_classes - 1, overlap_threshold=thresh_map)
    for n, imdb in enumerate(imdbs):
        for i in range(imdb.num_images):
            meta_info, im = mhd_reader(imdb.image_path_at(i).replace(".raw", ".mhd"))

            _t['im_detect'].tic()
            scores, boxes = im_detect(sess, net, im)
            _t['im_detect'].toc()

            _t['misc'].tic()

            # skip j = 0, because it's the background class
            for j in range(1, imdb.num_classes):
                inds = np.where(scores[:, j] > thresh_nms)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep, :]
                all_boxes[j][accu + i] = cls_dets

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][accu + i][:, -1] for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][accu + i][:, -1] >= image_thresh)[0]
                        all_boxes[j][accu + i] = all_boxes[j][accu + i][keep, :]
            
            # Accumulate AP
            preds = [np.reshape(all_boxes[j][accu + i], (-1, 5)) for j in range(1, imdb.num_classes)]
            preds = np.vstack(preds)
            pred_bb = preds[:, :4]
            pred_cls = [np.ones((len(all_boxes[j][accu + i]),)) * (j - 1) for j in range(1, imdb.num_classes)]
            pred_cls = np.hstack(pred_cls)
            pred_conf = preds[:, 4]
            gt_bb = imdb.roidb[i]["boxes"]
            gt_cls = imdb.roidb[i]["gt_classes"] - 1

            mAP.evaluate(pred_bb, pred_cls, pred_conf, gt_bb, gt_cls)
            _t['misc'].toc()

            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s\r'.format(accu + i + 1, 
                    num_images, _t['im_detect'].average_time, _t['misc'].average_time), end="")
        imdb_all_boxes = [[all_boxes[j][accu + i] for i in range(imdb.num_images)] for j in range(imdb.num_classes)]
        imdb.evaluate_detections(imdb_all_boxes, output_dir)
        accu += imdb.num_images

    # compute mAP
    precisions, recalls = mAP.compute_precision_recall_(0, True)
    AP = mAP.compute_ap(precisions, recalls)
    print("Evaluate AP: {:.3f}".format(AP))
    
    #det_file = os.path.join(output_dir, 'detections.pkl')
    #with open(det_file, 'wb') as f:
    #    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    #print('Evaluating detections')

