import torch
from torchinfo import summary
from pytorch3dunet.unet3d.model import UNet3D

def build_unet_model(device, state_dict=None, f_maps=32):
    # f_maps=32#16 for small GPU VRAM, 32 for large memory
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        final_sigmoid=False,   # for CrossEntropyLoss
        f_maps=f_maps,
        layer_order="gcr",
        # num_groups=8,
        is_segmentation=True,
    ).to(device)
    if state_dict is not None:
        mydict = torch.load(state_dict, map_location=device, weights_only=False)
        model.load_state_dict(mydict['model_state_dict'])
    return model