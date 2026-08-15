import numpy as np

def mri_layer_normalize(image_layer):
    '''Normalizes an MRI image layer (1 channel) to value range [0, 1] and converts to RGB format (3 channels).'''
    img_min, img_max = image_layer.min(), image_layer.max()
    norm_layer = (image_layer - img_min) / (img_max - img_min)
    rgb_layer = np.repeat(norm_layer[..., np.newaxis], 3, axis=2)
    return rgb_layer

def mri_normalize(mri):
    '''Normalizes an MRI volumeentirely to value range [0, 1]'''
    mri = np.maximum(mri, 0)
    mri_min, mri_max = np.min(mri), np.max(mri)
    normalized = (mri - mri_min) / (mri_max - mri_min)
    return normalized