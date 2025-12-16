## Input image size: 240x240x155
import os, json
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nibabel as nib
import cv2
from datetime import datetime
from scipy.ndimage import binary_opening

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

from sam_seg import get_predictor, get_mask_generator, sam_segmentation_with_bbox, show_segmentation_with_bbox, save_segmentation_with_bbox, sam_segmentation_with_point, show_segmentation_with_point, save_segmentation_with_point, sam_segmentation_raw, show_segmentation_raw, save_segmentation_raw
from medsam_seg import get_medsam_predictor, image_preprocessing, medsam, show_medsam_seg, save_medsam_seg
from evaluations import iou_score, dice_coefficient_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" LAYER OPERATIONS """

def _show_mri_layer(image, layer):
    '''
    Data (normalized image) Visualization (1 channel)
    
    :param image: Normalized MRI
    :param layer: layer #
    '''
    plt.style.use('default')
    plt.imshow(image[:, :, layer], cmap='gray', vmin=0, vmax=1)
    plt.show()

def _show_mri_rgb_layer(image, layer):
    '''
    Data (image) Visualization (3 channels)
    
    :param image: MRI
    :param layer: layer #
    '''
    plt.style.use('default')
    image_layer = image[:, :, layer]
    modified_layer = _mri_normalize_layer(image_layer)
    plt.imshow(modified_layer, vmin=0, vmax=1)
    plt.show()

def _save_mri_layer(image, layer, folder, image_name):
    '''
    Saves the grayscale MRI layer image
    
    :param image: Normalized MRI
    :param layer: layer #
    :param folder: Description
    :param image_name: Description
    '''
    plt.style.use('default')
    plt.imshow(image[:, :, layer], cmap='gray', vmin=0, vmax=1)
    os.makedirs(os.path.join('output', folder), exist_ok=True)
    plt.savefig(os.path.join('output', folder, f'{image_name}.png'))

def _mri_normalize_layer(image_layer):
    '''Normalizes an MRI image layer to value range [0, 1] and converts to RGB format (3 channels).'''
    img_min, img_max = image_layer.min(), image_layer.max()
    norm_layer = (image_layer - img_min) / (img_max - img_min)
    rgb_layer = np.repeat(norm_layer[..., np.newaxis], 3, axis=2)
    return rgb_layer

def run_sam_seg_layer_raw(mri, layer):
    '''
    Runs SAM on a single layer.
    '''
    mask_gen = get_mask_generator()
    image = _mri_normalize_layer(mri[:, :, layer])
    masks = sam_segmentation_raw(mask_gen, image)
    show_segmentation_raw(image, masks)
    # return masks

def run_sam_seg_layer_with_bbox(mri, layer, bbox):
    '''
    Runs SAM on a single layer and give a bounding box prompt.
    '''
    predictor = get_predictor()
    image = _mri_normalize_layer(mri[:, :, layer])
    mask = sam_segmentation_with_bbox(predictor, image, bbox)
    # show_segmentation_with_bbox(image, mask, bbox)
    return mask

def run_sam_seg_layer_with_point(mri, layer, point, label):
    '''
    Runs SAM on a single layer and give a bounding box prompt.
    '''
    predictor = get_predictor()
    image = _mri_normalize_layer(mri[:, :, layer])
    mask = sam_segmentation_with_point(predictor, image, point, label)
    return mask

def run_medsam_seg_layer(mri, layer, bbox):
    medsam_model = get_medsam_predictor(device)
    image = _mri_normalize_layer(mri[:, :, layer])
    img, box, h, w = image_preprocessing(image, bbox, device)
    mask = medsam(medsam_model, img, box, h, w)
    # show_medsam_seg(image, mask, bbox)
    return mask

def _get_bbox(image, margin=5, model='sam'):
    '''
    Docstring for _get_bbox
    
    :param image: Description
    :param margin: int, the margin/padding around bounding box
    :param model: 'sam' or 'medsam'
    '''
    if not isinstance(margin, int): raise TypeError(f"function _get_bbox param margin must be int, given {type(margin)}")

    denoised_image = binary_opening(image, structure=np.ones((2,2)))
    ys, xs = np.where(denoised_image)

    if len(xs) == 0:
        return None

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

""" MRI OPERATIONS """

def get_mri(path, ret_type:str='map'):
    '''
    Docstring for get_mri
    
    :param path: path to .nii / .nii.gz file
    :param ret_type: 'map' for a mapping of the mri file, or 'load' to load the mri directly to RAM
    '''
    if ret_type == 'map':
        return nib.load(path).dataobj
    elif ret_type == 'load':
        return nib.load(path).get_fdata()
    else:
        raise ValueError(f'ret_type must be \'map\' or \'load\', but given {ret_type}')

def _mri_normalize(mri):
    '''Normalizes an MRI entirely to value range [0, 1]'''
    mri_min, mri_max = np.min(mri), np.max(mri)
    normalized = (mri - mri_min) / (mri_max - mri_min)
    return normalized

def show_mri(mri):
    """
    Displays animated MRI.
    
    :param mri: normalized MRI as numpy array
    """
    fig, ax = plt.subplots()
    im = ax.imshow(mri[:, :, 0], cmap='gray', animated=True, vmin=0, vmax=1)
    title = ax.set_title("Slice 0")
    # title = ax.text(0.5, 1.05, "Slice 0",
    #                 ha='center', va='top',
    #                 transform=ax.transAxes,
    #                 animated=True)
    ax.axis('off')

    def update(frame):
        im.set_array(mri[:, :, frame])
        title.set_text(f"Slice {frame}")
        return [im, title]

    ani = FuncAnimation(
        fig,
        update,
        frames = mri.shape[2],
        interval=100,
        # blit=True
    )
    plt.show()

""" GROUND TRUTH OPERATIONS """

def get_ground_truth_mri(mri):
    pass

def get_ground_truth_layer(mri, layer):
    mri_image = mri[:, :, layer]
    return mri_image.astype(bool)

""" MAIN FUNCTIONS """

def sam_seg_main():
    ## Get runtime stamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    ## Change Variables
    parent_folder = 'data'
    instance_folder = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]
    # TODO: recursively loop through each instance folder until lowest level
    ground_truth_file = os.path.join(parent_folder, instance_folder, f'{instance_folder}-seg.nii.gz')
    flair_file = os.path.join(parent_folder, instance_folder, f'{instance_folder}-t2f.nii.gz')
    # TODO: if file path contains .nii.gz then use get_mri() for both gt_img and mri_img; 
    # TODO: if folder contains preprocessed_FLAIR.npy and preprocessed_segmentation.npy files then use np.load() directly
    gt_img = get_mri(ground_truth_file)
    mri_img = get_mri(flair_file)

    _, _, C = mri_img.shape

    dice_total = 0
    iou_total = 0
    segmented = 0
    dice_seg = 0
    iou_seg = 0
    scores = {'layers': {}}

    output_path = os.path.join(timestamp, instance_folder)

    for segment_layer in range(C):
        segment_bbox = _get_bbox(gt_img[:, :, segment_layer].astype(bool))
        if segment_bbox is not None:
            mask = run_sam_seg_layer_with_bbox(mri_img, segment_layer, segment_bbox)

            gt_mask = get_ground_truth_layer(gt_img, segment_layer)
            dice, iou = dice_coefficient_score(gt_mask, mask[0]), iou_score(gt_mask, mask[0])
            segmented += 1
            dice_total += dice
            iou_total += iou
            dice_seg += dice
            iou_seg += iou
            scores['layers'][segment_layer] = {'dice':dice, 'iou':iou}
            # print(f"Layer {segment_layerr} Dice Score: {dice}\n IoU Score: {iou}")

            # _save_mri_layer(gt_img, segment_layer, timestamp, f'ground_truth_layer{segment_layer}')
            layer_image = _mri_normalize_layer(mri_img[:, :, segment_layer])
            save_segmentation_with_bbox(layer_image, mask, gt_mask, segment_bbox, output_path, f'segmentation_layer{segment_layer}')
        else:
            dice_total += 1
            iou_total += 1
    
    scores['segmented'] = {'dice' : dice_seg / segmented, 'iou' : iou_seg / segmented}
    scores['final'] = {'dice' : dice_total / C, 'iou' : iou_total / C}
    with open(os.path.join('output', output_path, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)


def medsam_seg_main():
    ## Get runtime stamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    ## Change Variables
    parent_folder = 'data'
    instance_folder = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]
    # TODO: recursively loop through each instance folder until lowest level
    ground_truth_file = os.path.join(parent_folder, instance_folder, f'{instance_folder}-seg.nii.gz')
    flair_file = os.path.join(parent_folder, instance_folder, f'{instance_folder}-t2f.nii.gz')
    # TODO: if file path contains .nii.gz then use get_mri() for both gt_img and mri_img; 
    # TODO: if folder contains preprocessed_FLAIR.npy and preprocessed_segmentation.npy files then use np.load() directly
    gt_img = get_mri(ground_truth_file)
    mri_img = get_mri(flair_file)

    _, _, C = mri_img.shape

    dice_total = 0
    iou_total = 0
    segmented = 0
    dice_seg = 0
    iou_seg = 0
    scores = {'layers': {}}

    output_path = os.path.join(timestamp, instance_folder)

    for segment_layer in range(C):
        segment_bbox = _get_bbox(gt_img[:, :, segment_layer].astype(bool), model='medsam')
        if segment_bbox is not None:
            mask = run_medsam_seg_layer(mri_img, segment_layer, segment_bbox)
            # print(mask)
            layer_image = _mri_normalize_layer(mri_img[:, :, segment_layer])

            gt_mask = get_ground_truth_layer(gt_img, segment_layer)
            dice, iou = dice_coefficient_score(gt_mask, mask), iou_score(gt_mask, mask)
            segmented += 1
            dice_total += dice
            iou_total += iou
            dice_seg += dice
            iou_seg += iou
            scores['layers'][segment_layer] = {'dice':dice, 'iou':iou}
            # print(f"Dice Score: {dice}\n IoU Score: {iou}")

            # _save_mri_layer(gt_img, segment_layer, timestamp, f'ground_truth_layer{segment_layer}')
            save_medsam_seg(layer_image, mask, gt_mask, segment_bbox, output_path, f'medsam_segmentation_layer{segment_layer}')
        else:
            dice_total += 1
            iou_total += 1

    scores['segmented'] = {'dice' : dice_seg / segmented, 'iou' : iou_seg / segmented}
    scores['final'] = {'dice' : dice_total / C, 'iou' : iou_total / C}
    with open(os.path.join('output', output_path, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == '__main__':
    medsam_seg_main()
    