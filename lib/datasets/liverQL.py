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
from Liver_QL_Kits import get_mhd_list
from model.config import cfg

METType = {
    'MET_CHAR': np.char,
    'MET_SHORT': np.int16,
    'MET_LONG': np.int32,
    'MET_INT': np.int32,
    'MET_UCHAR': np.uint8,
    'MET_USHORT': np.uint16,
    'MET_ULONG': np.uint32,
    'MET_UINT': np.uint32,
    'MET_FLOAT': np.float32,
    'MET_FLOAT': np.float64
}


class liverQL(imdb):
    def __init__(self, image_set, year):
        imdb.__init__(self, 'liverQL_' + year + '_' + image_set)
        
        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'Liver_slices_' + self._image_set)
        self._classes = ('__background__', # background always index 0
            'liver',
        )
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.mhd'
        self._image_index, self._num_slices = get_mhd_list(osp.join(self._data_path, "mask"))
        #default to roidb handler
        self._roidb_handler = self.gt_roidb

    def image_path_at(self, i):
        """
        We have store the mask .mhd file path which means that we just need replace the 
        string `.mhd` by `.raw` and `mask` by 'liver'.
        """
        mask_mhd = self.roidb[i]['path']
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
        gt_roidb = [roi for roi in gt_roidb if roi['boxes'] is not None]
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

        meta_info, raw_image = self._mhd_reader(filepath)
        bbox = self._bbox_from_mask(raw_image)

        if bbox is not None:    # which means this is an empty mask image
            boxes[0,:] = bbox
            seg_areas[0] = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        else:
            boxes = None
            seg_areas = None
        cls_ = self._class_to_ind['liver']
        gt_classes[0] = cls_
        overlaps[0, cls_] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'path': filepath,
                'width': meta_info['DimSize'][1],
                'height': meta_info['DimSize'][0],
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _mhd_reader(self, rawpath):
        meta_info = {}
        # read .mhd file 
        with open(rawpath, 'r') as fmhd:
            for line in fmhd.readlines():
                parts = line.split()
                meta_info[parts[0]] = ' '.join(parts[2:])
        
        PrimaryKeys = ['NDims', 'DimSize', 'ElementType', 'ElementSpacing',
                        'ElementByteOrderMSB', 'ElementDataFile']
        for key in PrimaryKeys:
            if not key in meta_info:
                raise KeyError("Missing key `{}` in meta data of the mhd file".format(key))

        meta_info['NDims'] = int(meta_info['NDims'])
        meta_info['DimSize'] = [eval(ele) for ele in meta_info['DimSize'].split()]
        meta_info['ElementSpacing'] = [eval(ele) for ele in meta_info['ElementSpacing'].split()]
        meta_info['ElementByteOrderMSB'] = eval(meta_info['ElementByteOrderMSB'])

        raw_path = osp.join(osp.dirname(rawpath), meta_info['ElementDataFile'])

        # read .raw file
        with open(raw_path, 'rb') as fraw:
            buffer = fraw.read()
        
        raw_image = np.frombuffer(buffer, dtype=METType[meta_info['ElementType']])
        raw_image = np.reshape(raw_image, meta_info['DimSize'])

        return meta_info, raw_image 

    def _bbox_from_mask(self, mask):
        """ Calculate bounding box from a mask image 
        """
        bk_value = mask[0, 0]
        mask_pixels = np.where(mask > bk_value)
        if mask_pixels[0].size == 0:
            return None
        
        bbox = [
            np.min(mask_pixels[0]),
            np.min(mask_pixels[1]),
            np.max(mask_pixels[0]),
            np.max(mask_pixels[1])
        ]

        return bbox

if __name__ == '__main__':
    imdb = liverQL('train', '2018')
    for i in range(10):
        onecase = imdb.roidb[i]
        print(onecase)