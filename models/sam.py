import os

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

weights_path = {
    'vit_h' : os.path.join('model_pretrained', 'sam', 'sam_vit_h_4b8939.pth'),
    'vit_l' : os.path.join('model_pretrained', 'sam', 'sam_vit_l_0b3195.pth'),
    'vit_b' : os.path.join('model_pretrained', 'sam', 'sam_vit_b_01ec64.pth'),
    'default' : os.path.join('model_pretrained', 'sam', 'sam_vit_h_4b8939.pth'),
}

# SAM mask generator only uses numpy arrays
# SAM expects 3 channels RGB inputs

def get_predictor(model='vit_h'):
    sam = sam_model_registry[model](checkpoint=weights_path[model])
    predictor = SamPredictor(sam)
    return predictor

def get_mask_generator(model='vit_h'):
    sam = sam_model_registry[model](checkpoint=weights_path[model])
    # Generate Mask
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator