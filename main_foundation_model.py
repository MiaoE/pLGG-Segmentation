## Input image size: 240x240x155
import os, json
import argparse

# import gc
import numpy as np
import torch
# import cv2
from datetime import datetime
from scipy import ndimage
# from scipy.ndimage import binary_opening

from .models.sam import get_sam_predictor
from .models.medsam import get_medsam_predictor
from .data.loader import get_mri, get_gt_layer, load_image_and_gt, find_instances
from .data.preprocessing_steps import mri_layer_normalize
from .prediction.sam_predict import sam_segmentation_with_bbox, sam_segmentation_with_point
from .prediction.medsam_predict import medsam, medsam_image_preprocessing
from .prediction.sam_visualize import show_segmentation_with_bbox, save_segmentation_with_bbox, show_segmentation_with_point, save_segmentation_with_point
from .prediction.medsam_visualize import show_medsam_seg, save_medsam_seg
from training.evaluations import iou_score, dice_coefficient_score, hausdorff, hd95

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" LAYER WISE SEGMENTATION FUNCTIONS """

def run_sam_seg_layer_with_bbox(predictor, mri, layer, bbox):
    '''
    Runs SAM on a single layer and give a bounding box prompt.
    '''
    # predictor = get_predictor()
    image = mri_layer_normalize(mri[:, :, layer])
    mask = sam_segmentation_with_bbox(predictor, image, bbox)
    # show_segmentation_with_bbox(image, mask, bbox)
    return mask

def run_sam_seg_layer_with_point(mri, layer, point, label):
    '''
    Runs SAM on a single layer and give a bounding box prompt.
    '''
    predictor = get_sam_predictor()
    image = mri_layer_normalize(mri[:, :, layer])
    mask = sam_segmentation_with_point(predictor, image, point, label)
    return mask

def run_medsam_seg_layer(medsam_model, mri, layer, bbox):
    # medsam_model = get_medsam_predictor(device)
    image = mri_layer_normalize(mri[:, :, layer])
    img, box, h, w = medsam_image_preprocessing(image, bbox, device)
    mask = medsam(medsam_model, img, box, h, w)
    # show_medsam_seg(image, mask, bbox)
    return mask

""" HELPER for foundation model input prompt """
def get_bbox(image, margin=5, model='sam'):
    '''
    Gets the bounding box of a MRI layer (1 channel) given the ground truth segmentation mask
    
    :param image: ground truth segmentation mask
    :param margin: int, the margin/padding around bounding box
    :param model: 'sam' or 'medsam'
    '''
    if not isinstance(margin, int): raise TypeError(f"function _get_bbox param margin must be int, given {type(margin)}")

    # denoised_image = binary_opening(image, structure=np.ones((2,2)))
    labelled_image, num = ndimage.label(image)

    if num == 0:
        return None
    
    sizes = ndimage.sum(image, labelled_image, range(1, num+1))
    largest_label = np.argmax(sizes) + 1
    denoised_image = (labelled_image == largest_label)
    ys, xs = np.where(denoised_image)

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    if model == 'sam':
        return np.array([x_min-margin, y_min-margin, x_max+margin, y_max+margin])
    elif model == 'medsam':
        return np.array([[x_min-margin, y_min-margin, x_max+margin, y_max+margin]])
    else:
        raise ValueError(f'function _get_bbox param model must be sam or medsam, given {model}')

""" MAIN FUNCTIONS """

def sam_seg_main(parent_folder):
    ## Get runtime stamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    predictor = get_sam_predictor(model='vit_b')

    instance_folders = find_instances(parent_folder)
    dice_all = 0
    iou_all = 0
    hdist_all = 0
    hd95_all = 0
    count = 0
    num_layers = 0
    for instance, folder in instance_folders:
        mri_img, gt_img = load_image_and_gt(folder)

        _, _, C = mri_img.shape

        dice_total = 0
        iou_total = 0
        hdist_total = 0
        hd95_total = 0
        segmented = 0
        dice_seg = 0
        iou_seg = 0
        scores = {'layers': {}}

        output_path = os.path.join(f"{timestamp}-SAM", instance)
        os.makedirs(os.path.join('output', output_path), exist_ok=True)

        for segment_layer in range(C):
            segment_bbox = get_bbox(gt_img[:, :, segment_layer].astype(bool))
            if segment_bbox is not None:
                mask = run_sam_seg_layer_with_bbox(predictor, mri_img, segment_layer, segment_bbox)

                gt_mask = get_gt_layer(gt_img, segment_layer)
                dice, iou, hdist, hd95_val = dice_coefficient_score(gt_mask, mask), iou_score(gt_mask, mask), hausdorff(gt_mask, mask), hd95(gt_mask, mask)
                segmented += 1
                dice_total += dice
                iou_total += iou
                hdist_total += hdist
                hd95_total += hd95_val
                dice_seg += dice
                iou_seg += iou
                num_layers += 1
                scores['layers'][segment_layer] = {'dice':dice, 'iou':iou, 'hausdorff': hdist, 'hd95': hd95_val}
                # print(f"Layer {segment_layerr} Dice Score: {dice}\n IoU Score: {iou}")

                # _save_mri_layer(gt_img, segment_layer, timestamp, f'ground_truth_layer{segment_layer}')
                # layer_image = _mri_normalize_layer(mri_img[:, :, segment_layer])
                # save_segmentation_with_bbox(layer_image, mask, gt_mask, segment_bbox, output_path, f'segmentation_layer{segment_layer}')
            else:
                dice_total += 1
                iou_total += 1
        
        scores['segmented'] = {'dice' : dice_seg / segmented, 'iou' : iou_seg / segmented, 'hausdorff': hdist_total / segmented, 'hd95': hd95_total / segmented}
        scores['final'] = {'dice' : dice_total / C, 'iou' : iou_total / C, 'hausdorff': hdist_total / C, 'hd95': hd95_total / C}

        dice_all += dice_seg / segmented
        iou_all += iou_seg / segmented
        hdist_all += hdist_total / segmented
        hd95_all += hd95_total / segmented
        count += 1

        # Free RAM memory
        del mri_img
        del gt_img

        with open(os.path.join('output', output_path, 'scores.json'), 'w') as f:
            json.dump(scores, f, indent=4)
    results = {'DICE' : dice_all / count, 'IOU' : iou_all / count, 'Hausdorff' : hdist_all / count, 'HD95' : hd95_all / count, 'data_size' : count, 'num_layers_segmented' : num_layers}
    with open(os.path.join("output", f"{timestamp}-SAM", 'results.json'), 'w') as r:
        json.dump(results, r, indent=4)


def medsam_seg_main(parent_folder):
    ## Get runtime stamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    predictor = get_medsam_predictor(device)

    instance_folders = find_instances(parent_folder)
    dice_all = 0
    iou_all = 0
    hdist_all = 0
    hd95_all = 0
    count = 0
    num_layers = 0
    for instance, folder in instance_folders:
        mri_img, gt_img = load_image_and_gt(folder)

        _, _, C = mri_img.shape

        dice_total = 0
        iou_total = 0
        hdist_total = 0
        hd95_total = 0
        segmented = 0
        dice_seg = 0
        iou_seg = 0
        scores = {'layers': {}}

        output_path = os.path.join(f"{timestamp}-MedSAM", instance)
        os.makedirs(os.path.join('output', output_path), exist_ok=True)

        for segment_layer in range(C):
            segment_bbox = get_bbox(gt_img[:, :, segment_layer].astype(bool), model='medsam')
            if segment_bbox is not None:
                mask = run_medsam_seg_layer(predictor, mri_img, segment_layer, segment_bbox)
                # print(mask)
                layer_image = mri_layer_normalize(mri_img[:, :, segment_layer])

                gt_mask = get_gt_layer(gt_img, segment_layer)
                dice, iou, hdist, hd95_val = dice_coefficient_score(gt_mask, mask), iou_score(gt_mask, mask), hausdorff(gt_mask, mask), hd95(gt_mask, mask)
                segmented += 1
                dice_total += dice
                iou_total += iou
                hdist_total += hdist
                hd95_total += hd95_val
                dice_seg += dice
                iou_seg += iou
                num_layers += 1
                scores['layers'][segment_layer] = {'dice':dice, 'iou':iou, 'hausdorff': hdist, 'hd95': hd95_val}
                # print(f"Dice Score: {dice}\n IoU Score: {iou}")

                # _save_mri_layer(gt_img, segment_layer, timestamp, f'ground_truth_layer{segment_layer}')
                # save_medsam_seg(layer_image, mask, gt_mask, segment_bbox, output_path, f'medsam_segmentation_layer{segment_layer}')
            else:
                dice_total += 1
                iou_total += 1

        scores['segmented'] = {'dice' : dice_seg / segmented, 'iou' : iou_seg / segmented, 'hausdorff': hdist_total / segmented, 'hd95': hd95_total / segmented}
        scores['final'] = {'dice' : dice_total / C, 'iou' : iou_total / C, 'hausdorff': hdist_total / C, 'hd95': hd95_total / C}

        dice_all += dice_seg / segmented
        iou_all += iou_seg / segmented
        hdist_all += hdist_total / segmented
        hd95_all += hd95_total / segmented
        count += 1

        # Free RAM memory
        del mri_img
        del gt_img

        with open(os.path.join('output', output_path, 'scores.json'), 'w') as f:
            json.dump(scores, f, indent=4)
    results = {'DICE' : dice_all / count, 'IOU' : iou_all / count, 'Hausdorff' : hdist_all / count, 'HD95' : hd95_all / count, 'data_size' : count, 'num_layers_segmented' : num_layers}
    with open(os.path.join("output", f"{timestamp}-MedSAM", 'results.json'), 'w') as r:
        json.dump(results, r, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True, help="Path to root folder of patient data")
    parser.add_argument("-m", "--model", required=True, help="Foundation model (SAM or MedSAM)")
    args = parser.parse_args()
    folder = args.folder
    model = args.model

    if model.lower() == 'sam':
        sam_seg_main(folder)
    elif model.lower() == 'medsam':
        medsam_seg_main(folder)
    else:
        raise ValueError("Argument model is not SAM or MedSAM")
