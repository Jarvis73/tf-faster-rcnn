# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

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

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
    FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
    FONT = ImageFont.load_default()


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)

    return image


def draw_bounding_boxes(image, gt_boxes, im_info):
    num_boxes = gt_boxes.shape[0]
    gt_boxes_new = gt_boxes.copy()
    gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4].copy() / im_info[2])
    disp_image = Image.fromarray(np.uint8(image[0]))

    for i in range(num_boxes):
        this_class = int(gt_boxes_new[i, 4])
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes_new[i, 0],
                                      gt_boxes_new[i, 1],
                                      gt_boxes_new[i, 2],
                                      gt_boxes_new[i, 3],
                                      'N%02d-C%02d' % (i, this_class),
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    image[0, :] = np.array(disp_image)
    return image

def draw_bounding_boxes_with_prob(image, gt_boxes, prob, im_info):
    num_boxes = gt_boxes.shape[0]
    gt_boxes_new = gt_boxes.copy()
    gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4].copy() / im_info[2])
    disp_image = Image.fromarray(np.uint8(image[0]))

    for i in range(num_boxes):
        this_class = int(gt_boxes_new[i, 4])
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes_new[i, 0],
                                      gt_boxes_new[i, 1],
                                      gt_boxes_new[i, 2],
                                      gt_boxes_new[i, 3],
                                      'N%02d-C%02d-%.2f' % (i, this_class, prob[i]),
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    image[0, :] = np.array(disp_image)
    return image




def _mhd_reader(mhdpath):
    import os.path as osp
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



if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    # test_image_path = "C:/photos/tx.jpg"
    # image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)
    results_path = "C:/DataSet/LiverQL/Liver_slices_test/results_cls_liver.txt"
    with open(results_path, 'r') as f:
        while True:
            parts = f.readline().split()
            _, image = _mhd_reader(parts[0].replace("/home/jarvis", "C:").replace("mask", "liver").replace("_m_", "_o_"))
            min_val = np.min(image)
            max_val = np.max(image)
            image = (image - min_val) / (max_val - min_val) * 255
            image4D = np.zeros((1, image.shape[0], image.shape[1], 3), dtype=np.float32)
            for i in range(3):
                image4D[0,:,:,i] = image
            box = [int(eval(part)) for part in parts[2:]]
            box.append(1)
            prob = [float(parts[1])]
            gt_boxes = np.array(box).reshape((1, 5))
            im_info = (image.shape[0], image.shape[1], 1)
            dis = draw_bounding_boxes_with_prob(image4D, gt_boxes, prob, im_info)
            plt.imshow(dis[0])
            plt.show()
    
