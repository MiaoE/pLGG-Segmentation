import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .data.preprocessing_steps import mri_layer_normalize

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
    modified_layer = mri_layer_normalize(image_layer)
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