import os, json

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt

weights_path = {
    'vit_h' : os.path.join('model_pretrained', 'sam', 'sam_vit_h_4b8939.pth'),
    'vit_l' : os.path.join('model_pretrained', 'sam', 'sam_vit_l_0b3195.pth'),
    'vit_b' : os.path.join('model_pretrained', 'sam', 'sam_vit_b_01ec64.pth'),
    'default' : os.path.join('model_pretrained', 'sam', 'sam_vit_h_4b8939.pth'),
}

# SAM mask generator only uses numpy arrays
# SAM expects 3 channels RGB inputs

"""Get SAM models functions"""
def get_predictor(model='vit_h'):
    sam = sam_model_registry[model](checkpoint=weights_path[model])
    predictor = SamPredictor(sam)
    return predictor

def get_mask_generator(model='vit_h'):
    sam = sam_model_registry[model](checkpoint=weights_path[model])
    # Generate Mask
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

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

def _show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
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
def sam_segmentation_with_bbox(predictor, image, bbox):
    '''expects image to be 3 channels (W, H, 3)
    bbox in format [x,y,x,y] (single array)'''
    predictor.set_image(image)
    masks, _, _ = predictor.predict(box=bbox[None, :], multimask_output=False)
    '''output format
    [[[False False False ... False False False]
      [False False False ... False False False]
      [False False False ... False False False]
      ...
      [False False False ... False False False]
      [False False False ... False False False]
      [False False False ... False False False]]]'''
    return masks

def show_segmentation_with_bbox(image, mask, bbox):
    plt.imshow(image)
    _show_mask(mask[0], plt.gca())
    _show_box(bbox, plt.gca())
    plt.title(f"SAM Segmentation with Bounding Box Prompt")
    plt.axis('off')
    plt.show()

def save_segmentation_with_bbox(image, mask, bbox, folder, image_name):
    plt.imshow(image)
    _show_mask(mask[0], plt.gca())
    _show_box(bbox, plt.gca())
    plt.title(f"SAM Segmentation with Bounding Box Prompt")
    plt.axis('off')
    os.makedirs(os.path.join('output', folder), exist_ok=True)
    plt.savefig(os.path.join('output', folder, f'{image_name}.png'))

"""Point prompting SAM segmentation functions"""
def sam_segmentation_with_point(predictor, image, point, label):
    '''
    Docstring for sam_segmentation_with_point
    
    :param predictor: SAM predictor object
    :param image: the MRI layer image
    :param point: numpy array in [[x, y]] format (double array)
    :param label: numpy single element array, 1 - foreground point, 0 - background point 
    '''
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords=point, point_labels=label, multimask_output=False)
    return masks

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

"""Raw SAM segmentation functions"""
def sam_segmentation_raw(mask_generator, image):
    masks = mask_generator.generate(image)
    '''output format
    [{  'segmentation': array([[False,  True,  True, ...,  True, False, False],
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        ...,
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True]], shape=(240, 240)), 
        'area': 57574, 
        'bbox': [0, 0, 239, 239], 
        'predicted_iou': 1.0361067056655884, 
        'point_coords': [[236.25, 183.75]], 
        'stability_score': 0.9902777671813965, 
        'crop_box': [0, 0, 240, 240]
    }]
    '''
    return masks

def show_segmentation_raw(image, mask):
    plt.imshow(image)
    _show_anns(mask)
    plt.show()

def save_segmentation_raw(image, mask, folder, image_name):
    plt.imshow(image)
    _show_anns(mask)
    os.makedirs(os.path.join('output', folder), exist_ok=True)
    plt.savefig(os.path.join('output', folder, f'{image_name}.png'))