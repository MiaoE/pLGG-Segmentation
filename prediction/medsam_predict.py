import torch
import numpy as np
import torch.nn.functional as F
from skimage import io, transform

"""Helper function"""
def medsam_image_preprocessing(image, bbox, device):
    '''
    Preprocessing step done in the MedSAM paper
    
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

def medsam_predict(model, image, bbox, H, W):
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
