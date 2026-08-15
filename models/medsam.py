import os

from segment_anything import sam_model_registry

weights_path = {
    'lite' : os.path.join('model_pretrained', 'medsam', 'lite_medsam.pth'),
    'pt_prompt' : os.path.join('model_pretrained', 'medsam', 'medsam_point_prompt_flare22.pth'),
    'txt_prompt' : os.path.join('model_pretrained', 'medsam', 'medsam_text_prompt_flare22.pth'),
    'vit_b' : os.path.join('model_pretrained', 'medsam', 'medsam_vit_b.pth'),
    'default' : os.path.join('model_pretrained', 'medsam', 'medsam_vit_b.pth'),
}

def get_medsam_predictor(device, model='vit_b'):
    medsam = sam_model_registry[model](checkpoint=weights_path[model])
    medsam = medsam.to(device)
    medsam.eval()
    return medsam