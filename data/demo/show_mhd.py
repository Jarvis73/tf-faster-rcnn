# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhang Jianwei
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import cv2

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

def _mhd_reader(mhdpath):
    meta_info = {}
    # read .mhd file 
    with open(mhdpath, 'r') as fmhd:
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

    rawpath = osp.join(osp.dirname(mhdpath), meta_info['ElementDataFile'])

    # read .raw file
    with open(rawpath, 'rb') as fraw:
        buffer = fraw.read()
    
    raw_image = np.frombuffer(buffer, dtype=METType[meta_info['ElementType']])
    raw_image = np.reshape(raw_image, meta_info['DimSize'])

    return meta_info, raw_image.copy()

def show_mhd(path, width=400, level=60, outer_val=-1000):
    meta_info, rawimage = _mhd_reader(path)
    # low_val = level - width / 2
    # high_val = level + width / 2
    # rawimage[rawimage < low_val] = low_val
    # rawimage[rawimage > high_val] = high_val

    low = np.min(rawimage)
    high = np.max(rawimage)
    rawimage = (rawimage - low) / (high - low) * 255
    rawimage = rawimage.astype(np.int32)
    #print(rawimage[246:266, 246:264])
    #print(np.bincount(rawimage.flat))
    # cv2.imshow("Show medical image", rawimage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(rawimage, cmap='gray')
    plt.show()

if __name__ == '__main__':
    path = "C:/DataSet/LiverQL/Liver_slices_train/liver/P002_o_21.mhd"
    show_mhd(path)
