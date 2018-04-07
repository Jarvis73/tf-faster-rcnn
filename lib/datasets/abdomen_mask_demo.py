import os
import os.path as osp
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from Liver_Kits import mhd_reader
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from multiprocessing import Pool

def abdomen_mask_demo(mhdfile):
    meta_info, raw_image = mhd_reader(src)
    im = np.reshape(raw_image, (512, 512))

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.flatten()
    # Step 1: Convert into a binary image.
    binary = im > -400
    ax[0].imshow(binary, cmap='gray')
    ax[0].set_title("threshold")
    # Step 1.5: fill border pixels
    binary[:,[0,-1]] = False
    binary[[0,-1],:] = False
    # Step 2: Remove the blobs connected to the border of the image.
    binary = clear_border(binary)
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title("cleared")
    # Step 3: Fill holes inside the binary mask image
    binary = binary_fill_holes(binary)
    ax[2].imshow(binary, cmap='gray')
    ax[2].set_title("fill holes")
    # Step 4: remove other parts, only keep abdomen
    labeled = label(binary)
    ax[3].imshow(labeled, cmap='gray')
    ax[3].set_title("label_image")
    # Step 5: Keep the labels with the largest areas
    areas = [r.area for r in regionprops(labeled)]
    areas.sort()
    if len(areas) > 1:
        for region in regionprops(labeled):
            if region.area < areas[-1]:
                for coords in region.coords:
                    labeled[coords[0], coords[1]] = 0

    for i in range(4):
        ax[i].axis('off')
    ax[4].imshow(labeled, cmap='gray')
    ax[4].set_title("1_labels")

    plt.tight_layout(w_pad=0.5, h_pad=0.5)
    plt.show()

def extract_mask(src, dst, verbose=False):
    if verbose:
        print(src, end='\t')
    meta_info, raw_image = mhd_reader(src)
    im = np.reshape(raw_image, (512, 512))

    binary = im > -400
    binary[:,[0,-1]] = False
    binary[[0,-1],:] = False
    binary = clear_border(binary)
    binary = binary_fill_holes(binary)
    labeled = label(binary)
    areas = [r.area for r in regionprops(labeled)]
    areas.sort()
    if len(areas) > 1:
        for region in regionprops(labeled):
            if region.area < areas[-1]:
                for coords in region.coords:
                    labeled[coords[0], coords[1]] = 0

    #plt.imshow(labeled, cmap='gray')
    #plt.imshow(binary, cmap='gray')
    #plt.axis('off')
    #plt.savefig(dst)
    summ = np.sum(labeled)
    print(summ)
    if summ < 10000:
        raise ValueError

def generate_mask(srcDir, dstDir):
    mhdlist = glob(osp.join(srcDir, "*.mhd"))
    outlist = [osp.join(dstDir, osp.basename(mhdfile).replace(".mhd", ".jpg")) for mhdfile in mhdlist]
    with Pool(4) as p:
        p.starmap(extract_mask, zip(mhdlist, outlist, [True] * len(mhdlist)))
    


if __name__ == '__main__':
    srcDir = "D:/DataSet/LiverQL/Liver_2017_train/liver"
    dstDir = "D:/DataSet/LiverQL/Liver_2017_train/abdomen"
    generate_mask(srcDir, dstDir)