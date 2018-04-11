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
import matplotlib.pyplot as plt
from matplotlib import patches 
import os.path as osp
import sys
ds_path = osp.join(osp.dirname(__file__), "..", "datasets")
sys.path.insert(0, ds_path)
from Liver_Kits import mhd_reader    

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

def draw_bounding_boxes_with_plt(image, pred_boxes, gt_boxes, probs):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap='gray')
    for i in range(pred_boxes.shape[0]):
        bbox = pred_boxes[i, :]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                 linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[3], "C{:d}-{:.2f}".format(bbox[4], probs[i]), 
                 style="normal", weight="bold", size=7, color='w', 
                 bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 1})

    for i in range(gt_boxes.shape[0]):
        bbox = gt_boxes[i, :]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                 linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def get_line(f):
    parts = f.readline().split()
    path = parts[0]
    prob = float(parts[1])
    bbox = [int(eval(ele)) for ele in parts[2:]]
    bbox.append(1)
    return path, prob, bbox

if __name__ == '__main__':
    import sys
    sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
    from datasets.Liver_Kits import bbox_from_mask_2D
    results_path = "D:/DataSet/LiverQL/Liver_2017_test/results_cls_liver.txt"
    with open(results_path, 'r') as f:
        path, prob, bbox = get_line(f)
        tpath = path
        while True:
            boxes = []
            probs = []
            while tpath == path:
                boxes.append(bbox)
                probs.append(prob)
                tpath, prob, bbox = get_line(f)
            print(path)
            _, image = mhd_reader(path.replace("/home/jarvis", "D:").replace("mask", "liver").replace("_m_", "_o_"))
            _, mask = mhd_reader(path.replace("/home/jarvis", "D:"))
            gt_boxes = bbox_from_mask_2D(mask)
            gt_boxes.append(1)
            gt_boxes = np.array(gt_boxes).reshape((-1, 5))
            pred_boxes = np.array(boxes).reshape((-1, 5))
            draw_bounding_boxes_with_plt(image, pred_boxes, gt_boxes, probs)
            plt.show()
            path = tpath
    
