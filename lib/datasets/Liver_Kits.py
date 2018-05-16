import os
import os.path as osp
from glob import glob
import numpy as np
import pickle
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.feature import canny
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

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

def extract_slices(SrcDir, DstDir_o, DstDir_m):
    SrcDirs = os.listdir(SrcDir)

    for src in SrcDirs:
        src_liver = SrcDir + src + "/Liver.vol"
        src_liver_mask = SrcDir + src + "/Liver.mask.vol"

        dst_liver = DstDir_o + src + "_o"
        dst_liver_mask = DstDir_m + src + "_m"

        worker_path = osp.join(osp.dirname(__file__), "$VolConverter.exe")

        os.system(worker_path + " 1 " + src_liver + " " + dst_liver)
        os.system(worker_path + " 1 " + src_liver_mask + " " +  dst_liver_mask)

def mhd_reader(mhdpath, only_meta=False):
    meta_info = {}
    # read .mhd file 
    with open(mhdpath, 'r') as fmhd:
        for line in fmhd.readlines():
            parts = line.split()
            meta_info[parts[0]] = ' '.join(parts[2:])
    
    PrimaryKeys = ['NDims', 'DimSize', 'ElementType', 'ElementSpacing', 'ElementDataFile']
    for key in PrimaryKeys:
        if not key in meta_info:
            raise KeyError("Missing key `{}` in meta data of the mhd file".format(key))

    meta_info['NDims'] = int(meta_info['NDims'])
    meta_info['DimSize'] = [eval(ele) for ele in meta_info['DimSize'].split()]
    meta_info['ElementSpacing'] = [eval(ele) for ele in meta_info['ElementSpacing'].split()]
    #meta_info['ElementByteOrderMSB'] = eval(meta_info['ElementByteOrderMSB'])

    raw_image = None
    if not only_meta:
        rawpath = osp.join(osp.dirname(mhdpath), meta_info['ElementDataFile'])

        # read .raw file
        with open(rawpath, 'rb') as fraw:
            buffer = fraw.read()
    
        raw_image = np.frombuffer(buffer, dtype=METType[meta_info['ElementType']])
        raw_image = np.reshape(raw_image, meta_info['DimSize'])

    return meta_info, raw_image 

def raw_reader(raw_path, element_type="MET_SHORT", dim_size=[512, 512]):
    with open(raw_path, "rb") as fraw:
        buffer = fraw.read()
    
    raw_image = np.frombuffer(buffer, dtype=METType[element_type])
    raw_image = np.reshape(raw_image, dim_size)

    return raw_image

def bbox_from_mask_2D(mask, bk_value=None):
    """ Calculate bounding box from a 2D mask image 
    """
    if bk_value is None:
        bk_value = mask[0, 0]
    mask_pixels = np.where(mask > bk_value)
    if mask_pixels[0].size == 0:
        return None
    
    bbox = [
        np.min(mask_pixels[1]),
        np.min(mask_pixels[0]),
        np.max(mask_pixels[1]),
        np.max(mask_pixels[0])
    ]

    return bbox

def bbox_from_mask_3D(mask, bk_value=None):
    """ Calculate bounding box from a 2D mask image 

    Params
    ------
    `mask`: 3D ndarray, mask image with shape [depth, height, width]
    """
    if bk_value is None:
        bk_value = mask[0, 0, 0]
    mask_pixels = np.where(mask > bk_value)
    if mask_pixels[0].size == 0:
        return None
    
    bbox = [
        np.min(mask_pixels[2]),
        np.min(mask_pixels[1]),
        np.min(mask_pixels[0]),
        np.max(mask_pixels[2]),
        np.max(mask_pixels[1]),
        np.max(mask_pixels[0])
    ]

    return bbox

def abdominal_mask(image, low_val=-400):
    binary = image > low_val
    binary[:,[0,-1]] = False
    binary[[0,-1],:] = False
    binary = binary_fill_holes(clear_border(binary))
    labeled = label(binary)
    areas = [r.area for r in regionprops(labeled)]
    areas.sort()
    if len(areas) > 1:
        for region in regionprops(labeled):
            if region.area < areas[-1]:
                for coords in region.coords:
                    labeled[coords[0], coords[1]] = 0

    return labeled

def get_mhd_list(SrcDir):
    if not osp.exists(SrcDir):
        raise FileNotFoundError("{} can not found!".format(SrcDir))
        
    mhd_list = glob(osp.join(SrcDir, "*.mhd"))
    size = len(mhd_list)
    return mhd_list, size

def get_mhd_list_with_liver(SrcDir, threshold=0, verbose=False):
    """ SrcDir should be a mask dir """
    cache_file = osp.join(SrcDir, "liver_slices_%d.pkl" % threshold)
    if osp.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            try:
                keep_mhd_list = pickle.load(fid)
            except:
                keep_mhd_list = pickle.load(fid, encoding='bytes')
        print("mhd list loaded from {}".format(cache_file))
        return keep_mhd_list
    

    all_mhd_list, all_mhd_length = get_mhd_list(SrcDir)
    keep_mhd_list = []
    for mhdfile in all_mhd_list:
        if verbose:
            print(mhdfile)
        _, raw = mhd_reader(mhdfile)
        # bbox = bbox_from_mask_2D(raw)
        area = np.sum(raw > 0)
        if area > threshold:
            keep_mhd_list.append(mhdfile)
    
    with open(cache_file, 'wb') as fid:
        pickle.dump(keep_mhd_list, fid, pickle.HIGHEST_PROTOCOL)
    print("Write mhd list to {}".format(cache_file))

    return keep_mhd_list

def get_canny(image, sigma=3, **kwargs):
    result = canny(image, sigma=sigma, **kwargs)
    return result

if __name__ == '__main__':
    SrcDir = "C:/DataSet/LiverQL/Liver-Ref/"
    SrcDir_o = "C:/DataSet/LiverQL/Liver_slices_train/liver/"
    SrcDir_m = "C:/DataSet/LiverQL/Liver_slices_train/mask/"

    if False:
        extract_slices(SrcDir, SrcDir_o, SrcDir_m)
    
    if False:
        out_file = "D:/DataSet/LiverQL/areas.txt"
        
        if not osp.exists(out_file):
            SrcDir_m1 = "D:/DataSet/LiverQL/Liver_2016_train/mask/"
            SrcDir_m2 = "D:/DataSet/LiverQL/Liver_2017_train/mask/"
            SrcDir_m3 = "D:/DataSet/LiverQL/Liver_2017_test/mask/"
            list1 = get_mhd_list_with_liver(SrcDir_m1, 2000, False)
            list2 = get_mhd_list_with_liver(SrcDir_m2, 2000, False)
            mask_list = list1 + list2
            areas = []
            with open(out_file, "w") as f:
                for i, mask_file in enumerate(mask_list):
                    _, mask = mhd_reader(mask_file)
                    area = np.sum(mask > 0)
                    areas.append(area)
                    f.write("%d\n" % area)
                    if i % 100 == 0:
                        print(i)
        else:
            with open(out_file, "r") as f:
                lines = f.readlines()
            areas = [int(line) for line in lines]
        
        plt.hist(areas, 25)
        plt.show()

    if False:
        SrcDir_m1 = "D:/DataSet/LiverQL/Liver_2016_train/mask/"
        SrcDir_m2 = "D:/DataSet/LiverQL/Liver_2017_train/mask/"
        SrcDir_m3 = "D:/DataSet/LiverQL/Liver_2017_test/mask/"
        list1 = get_mhd_list_with_liver(SrcDir_m1, False)
        list2 = get_mhd_list_with_liver(SrcDir_m2, False)
        list3 = get_mhd_list_with_liver(SrcDir_m3, False)
        mask_list = list1 + list2
        mask_list2 = list3
        for mask_file in mask_list2:
            _, mask = mhd_reader(mask_file)
            area = np.sum(mask > 0)
            if area < 2000:
                plt.imshow(mask)
                plt.show()

    if False:
        path = "D:/DataSet/LiverQL/Liver_2018_train/liver/R001_o_26.mhd"
        _, image = mhd_reader(path)
        labels = abdominal_mask(image)
        print(np.bincount(labels.flat))
        print(labels.dtype)
        print(np.max(labels), np.min(labels))

    if False:
        # check abdomen window (width and level)
        path = "D:/DataSet/LiverQL/Liver_2018_train/liver/R001_o_26.mhd"
        _, image = mhd_reader(path)
        image = (np.clip(image, 55 - 125, 55 + 125) - (55 - 125)) / 2**16 * 250
        plt.hist(image.flat, 20)
        #plt.imshow(image, cmap="gray")
        plt.show()

    if False:
        path = "C:/DataSet/LiverQL/Liver_2017_train/liver/Q001_o_57.mhd"
        _, image = mhd_reader(path)
        image = (np.clip(image, -300 - 700, -300 + 700) - (-300 - 700)) / 2**16 * 1400
        #image = image.astype(np.int32)
        plt.imshow(image, cmap="gray")
        plt.show()
