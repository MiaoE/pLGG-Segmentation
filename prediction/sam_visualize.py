import os

import numpy as np
import matplotlib.pyplot as plt

# SAM mask generator only uses numpy arrays
# SAM expects 3 channels RGB inputs

"""Helper plotting/image displaying functions"""
def _show_anns(anns):
    if len(anns) == 0:
        return
    # sort by largest area first
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def _show_mask(mask, ax, random_color=False, gt=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif gt:
        color = np.array([255/255, 255/255, 0/255, 0.25])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    mask_image = mask[:, :, None] * color[None, None, :]
    ax.imshow(mask_image)

def _show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def _show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1)
    

"""Bounding box prompting SAM segmentation functions"""
def show_segmentation_with_bbox(image, mask, gt_mask, bbox):
    plt.imshow(image)
    _show_mask(mask, plt.gca())
    _show_mask(gt_mask, plt.gca(), gt=True)
    _show_box(bbox, plt.gca())
    plt.title(f"SAM Segmentation with Bounding Box Prompt")
    plt.axis('off')
    plt.show()

def save_segmentation_with_bbox(image, mask, gt_mask, bbox, folder, image_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    _show_mask(mask, plt.gca())
    _show_mask(gt_mask, plt.gca(), gt=True)
    _show_box(bbox, plt.gca())
    ax.set_title(f"SAM Segmentation with Bounding Box Prompt")
    ax.axis('off')
    os.makedirs(os.path.join('output', folder), exist_ok=True)
    fig.savefig(os.path.join('output', folder, f'{image_name}.png'))
    plt.close(fig)

"""Point prompting SAM segmentation functions"""
def show_segmentation_with_point(image, mask, point, label):
    plt.imshow(image)
    _show_mask(mask[0], plt.gca())
    _show_points(point, label, plt.gca())
    plt.title(f"SAM Segmentation with Point Prompt")
    plt.axis('off')
    plt.show()

def save_segmentation_with_point(image, mask, point, label, folder, image_name):
    plt.imshow(image)
    _show_mask(mask[0], plt.gca())
    _show_points(point, label, plt.gca())
    plt.title(f"SAM Segmentation with Point Prompt")
    plt.axis('off')
    os.makedirs(os.path.join('output', folder), exist_ok=True)
    plt.savefig(os.path.join('output', folder, f'{image_name}.png'))

""" SAM segmentation mask functions"""
def show_segmentation_mask(image, mask):
    plt.imshow(image)
    _show_anns(mask)
    plt.show()

def save_segmentation_mask(image, mask, folder, image_name):
    plt.imshow(image)
    _show_anns(mask)
    os.makedirs(os.path.join('output', folder), exist_ok=True)
    plt.savefig(os.path.join('output', folder, f'{image_name}.png'))