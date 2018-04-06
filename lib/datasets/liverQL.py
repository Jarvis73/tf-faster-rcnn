# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhang Jianwei
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pickle
import numpy as np
import scipy

if __name__ == '__main__':
    import sys
    lib_path = osp.join(osp.dirname(__file__), "..")
    sys.path.insert(0, lib_path)
from datasets.imdb import imdb
from datasets.Liver_Kits import get_mhd_list_with_liver, mhd_reader, bbox_from_mask
from model.config import cfg


class liverQL(imdb):
    def __init__(self, image_set, year):
        imdb.__init__(self, 'liverQL_' + year + '_' + image_set)
        
        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'Liver_{}_{}'.format(self._year, self._image_set))
        self._classes = ('__background__', # background always index 0
            'liver',
        )
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.mhd'
        self._image_index = get_mhd_list_with_liver(osp.join(self._data_path, "mask"))

        #default to roidb handler
        self._roidb_handler = self.gt_roidb

    def image_path_at(self, i):
        """
        We have store the mask .mhd file path which means that we just need replace the 
        string `.mhd` by `.raw` and `mask` by 'liver'.
        """
        mask_mhd = self.image_index[i]
        mask_dir = osp.dirname(mask_mhd)
        mask_mhd_basename = osp.basename(mask_mhd)
        liver_slices_dir = osp.dirname(mask_dir)
        liver_raw_basename = mask_mhd_basename.replace('_m_', '_o_').replace('.mhd', '.raw')
        liver_raw = osp.join(liver_slices_dir, "liver", liver_raw_basename)
        return liver_raw

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_LiverQL_annotation(fpath) for fpath in self.image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote gt roidb to {}'.format(cache_file))

        return gt_roidb
    
    def _load_LiverQL_annotation(self, filepath):
        """ Load image and bounding boxes info from mhd/raw file in the LiverQL dataset.
        """
        boxes = np.zeros((1, 4), dtype=np.uint16)
        gt_classes = np.zeros((1), dtype=np.int32)
        overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((1), dtype=np.float32)

        meta_info, raw_image = mhd_reader(filepath)
        bbox = bbox_from_mask(raw_image)

        if bbox is not None:    # which means this is an empty mask image
            boxes[0,:] = bbox
            seg_areas[0] = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        else:
            boxes = None
            seg_areas = None
            print(filepath)
        cls_ = self._class_to_ind['liver']
        gt_classes[0] = cls_
        overlaps[0, cls_] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'width': meta_info['DimSize'][1],
                'height': meta_info['DimSize'][0],
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _write_liverQL_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} liverQL results file'.format(cls))
            filename = osp.join(self._data_path, 'results_cls_{}.txt'.format(cls))
            with open(filename, 'w') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets.size == 0:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_liverQL_results_file(all_boxes)

        # compute IOU
        total_iou = []
        iou_file = osp.join(self._data_path, 'iou.txt')
        f = open(iou_file, 'w')
        for cls_ind in range(1, self.num_classes):
            for im_ind in range(self.num_images):
                dets = all_boxes[cls_ind][im_ind]
                if dets.size == 0:
                    total_iou.append(0.)
                    f.write("0.000 no bbox\n")
                    continue
                min_ = np.min(dets, axis=0)
                max_ = np.max(dets, axis=0)
                pred_bbox = np.array([
                    min_[0], min_[1], max_[2], max_[3]
                ])
                gt_bbox = self.roidb[im_ind]['boxes'][0]
                
                x1 = np.maximum(pred_bbox[0], gt_bbox[0])
                y1 = np.maximum(pred_bbox[1], gt_bbox[1])
                x2 = np.minimum(pred_bbox[2], gt_bbox[2])
                y2 = np.minimum(pred_bbox[3], gt_bbox[3])
                if x1 >= x2 or y1 >= y2:
                    total_iou.append(0.)
                    f.write("0.000 no overlap\n")
                else:
                    area1 = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
                    area2 = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
                    area3 = (x2 - x1 + 1) * (y2 - y1 + 1)
                    iou = area3 / (area1 + area2 - area3)
                    total_iou.append(iou)
                    f.write("%.3f\n" % iou)

        avg_iou = np.mean(np.array(total_iou))
        print("Mean iou {:.3f} on {:d} images".format(avg_iou, self.num_images))
        f.close()


if __name__ == '__main__':
    imdb = liverQL('train', '2018')
    for i in range(10):
        onecase = imdb.roidb[i]
        print(onecase)