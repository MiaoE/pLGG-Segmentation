import os

import numpy as np
import matplotlib.pyplot as plt

def _show_mask(mask, ax, random_color=False, gt=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif gt:
        color = np.array([255/255, 255/255, 0/255, 0.25])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def _show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
def show_medsam_seg(image, mask, gt_mask, bbox):
    '''
    Docstring for show_medsam_seg
    
    :param image: Original image (3 channels)
    :param mask: MedSAM segmentation result
    :param bbox: Original bounding box numpy array
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    _show_box(bbox[0], plt.gca())
    _show_mask(mask, plt.gca())
    _show_mask(gt_mask, plt.gca(), gt=True)
    ax.set_title("MedSAM Segmentation with Bounding Box")
    plt.show()

def save_medsam_seg(image, mask, gt_mask, bbox, folder, image_name):
    '''
    Docstring for show_medsam_seg
    
    :param image: Original image (3 channels)
    :param mask: MedSAM segmentation result
    :param bbox: Original bounding box numpy array
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    _show_box(bbox[0], plt.gca())
    _show_mask(mask, plt.gca())
    _show_mask(gt_mask, plt.gca(), gt=True)
    ax.set_title("MedSAM Segmentation with Bounding Box")
    
    os.makedirs(os.path.join('output', folder), exist_ok=True)
    fig.savefig(os.path.join('output', folder, f'{image_name}.png'))
    plt.close(fig)