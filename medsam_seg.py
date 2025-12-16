import os

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import io, transform

weights_path = {
    'lite' : os.path.join('model_pretrained', 'medsam', 'lite_medsam.pth'),
    'pt_prompt' : os.path.join('model_pretrained', 'medsam', 'medsam_point_prompt_flare22.pth'),
    'txt_prompt' : os.path.join('model_pretrained', 'medsam', 'medsam_text_prompt_flare22.pth'),
    'vit_b' : os.path.join('model_pretrained', 'medsam', 'medsam_vit_b.pth'),
    'default' : os.path.join('model_pretrained', 'medsam', 'medsam_vit_b.pth'),
}

"""Get MedSAM models functions"""
def get_medsam_predictor(device, model='vit_b'):
    medsam = sam_model_registry[model](checkpoint=weights_path[model])
    medsam = medsam.to(device)
    medsam.eval()
    return medsam

"""Helper plotting/image displaying functions"""

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

"""Helper function"""
def image_preprocessing(image, bbox, device):
    '''
    Docstring for image_preprocessing
    
    :param image: 3 channel medical image
    :param bbox: numpy array bounding box in [[x,y,x,y]] (double array)
    '''
    H, W, _ = image.shape

    img_1024 = transform.resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # transfer box_np t0 1024x1024 scale
    box_1024 = bbox / np.array([W, H, W, H]) * 1024
    return (img_1024_tensor, box_1024, H, W)

"""Bounding box prompting MedSAM inference function"""
def _medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def medsam(model, image, bbox, H, W):
    '''
    Docstring for medsam
    
    :param model: medsam model
    :param image: image tensor (from preprocessing)
    :param bbox: bounding box numpy array (from preprocessing)
    :param H: original image height (from preprocessing)
    :param W: original image width (from preprocessing)
    '''
    with torch.no_grad():
        image_embedding = model.image_encoder(image)
    medsam_seg = _medsam_inference(model, image_embedding, bbox, H, W)
    return medsam_seg

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