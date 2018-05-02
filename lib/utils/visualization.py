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
ds_path = osp.join(osp.dirname(__file__), "..")
sys.path.insert(0, ds_path)
from datasets.Liver_Kits import mhd_reader, get_mhd_list_with_liver
from model.config import cfg
import pickle

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

def draw_bounding_boxes_with_plt(image, pred_boxes, gt_boxes, probs, path=None, show=True):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap='gray')
    for i in range(pred_boxes.shape[0]):
        bbox = pred_boxes[i, :]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                 linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        if probs[i]:
            ax.text(bbox[0], bbox[3], "C{:d}-{:.2f}".format(bbox[4], probs[i]), 
                     style="normal", weight="bold", size=7, color='w', 
                     bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
        else:
            ax.text(bbox[0], bbox[3], "C{:d}".format(bbox[4]), 
                     style="normal", weight="bold", size=7, color='w', 
                     bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 1})

    for i in range(gt_boxes.shape[0]):
        bbox = gt_boxes[i, :]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                 linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.axis("off")
    if show:
        plt.show()
    else:
        plt.savefig(path.replace("mask", "prediction").replace(".mhd", ".png").replace("_m_", "_p_").replace("/home/jarvis", "D:"))
    plt.close()

def get_line(f):
    parts = f.readline().split()
    if parts:
        path = parts[0]
        prob = float(parts[1])
        bbox = [int(eval(ele)) for ele in parts[2:]]
        bbox.append(1)
        return path, prob, bbox
    else:
        return None, None, None

def draw_threshold_iou_curve(filename, a=0, b=1, step=0.01):
    all_boxes = []
    all_probs = []
    all_paths = []
    with open(filename, "r") as f:
        path, prob, bbox = get_line(f)
        tpath = path
        while True:
            boxes = []
            probs = []
            while tpath == path:
                boxes.append(bbox)
                probs.append(prob)
                tpath, prob, bbox = get_line(f)
            all_boxes.append(np.array(boxes))
            all_probs.append(np.array(probs))
            all_paths.append(path)
            path = tpath
            if path is None:
                break
    
    # extract gt_boxes
    # image_index = get_mhd_list_with_liver(osp.join(cfg.DATA_DIR, "Liver_2017_test", "mask"))
    cache_file = osp.join(cfg.DATA_DIR, "cache", 'liverQL_2017_test_gt_roidb_%d.pkl' % cfg.MASK_AREA_LO)
    if osp.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            try:
                roidb = pickle.load(fid)
            except:
                roidb = pickle.load(fid, encoding='bytes')
    else:
        raise NotImplementedError

    thresholds = np.arange(a, b, step)
    mean_ious = []
    for threshold in thresholds:
        total_iou = []
        for i in range(len(all_probs)):
            keep = np.where(all_probs[i] > threshold)[0]
            if keep.size == 0:
                keep = np.argmax(all_probs[i])
            dets = all_boxes[i][[keep]]
            min_ = np.min(dets, axis=0)
            max_ = np.max(dets, axis=0)

            pred_bbox = np.array([min_[0], min_[1], max_[0], max_[1]])
            gt_bbox = roidb[i]["boxes"][0]
            
            x1 = np.maximum(pred_bbox[0], gt_bbox[0])
            y1 = np.maximum(pred_bbox[1], gt_bbox[1])
            x2 = np.minimum(pred_bbox[2], gt_bbox[2])
            y2 = np.minimum(pred_bbox[3], gt_bbox[3])
            if x1 >= x2 or y1 >= y2:
                total_iou.append(0.)
            else:
                area1 = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
                area2 = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
                area3 = (x2 - x1 + 1) * (y2 - y1 + 1)
                iou = area3 / (area1 + area2 - area3)
                total_iou.append(iou)
        mean_iou = np.mean(total_iou)
        print(mean_iou)
        mean_ious.append(mean_iou)             

    plt.plot(thresholds, mean_ious)
    plt.show()


if __name__ == '__main__':
    results_path = "C:/DataSet/LiverQL/Liver_2017_test/results_cls_liver.txt"
    #results_path = "D:/DataSet/LiverQL/Liver_2016_train/results_cls_liver.txt"
    if True:
        import sys
        sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
        from datasets.Liver_Kits import bbox_from_mask_2D
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
                _, image = mhd_reader(path.replace("/home/jarvis", "C:").replace("mask", "liver").replace("_m_", "_o_"))
                _, mask = mhd_reader(path.replace("/home/jarvis", "C:"))
                gt_boxes = bbox_from_mask_2D(mask)
                gt_boxes.append(1)
                gt_boxes = np.array(gt_boxes).reshape((-1, 5))
                pred_boxes = np.array(boxes).reshape((-1, 5))
                draw_bounding_boxes_with_plt(image, pred_boxes, gt_boxes, probs, path, show=True)
                path = tpath
        
    if False:
        draw_threshold_iou_curve(results_path)