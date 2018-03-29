import os
import os.path as osp
from glob import glob

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

def get_mhd_list(SrcDir):
    if not osp.exists(SrcDir):
        raise FileNotFoundError("{} can not found!".format(SrcDir))
        
    mhd_list = glob(osp.join(SrcDir, "*.mhd"))
    size = len(mhd_list)
    return mhd_list, size

if __name__ == '__main__':
    if False:
        SrcDir = "../Liver-Ref/"
        DstDir_o = "../Liver_slices/liver/"
        DstDir_m = "../Liver_slices/mask/"
        extract_slices(SrcDir, DstDir_o, DstDir_m)

