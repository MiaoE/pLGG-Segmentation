import os

import nibabel as nib
import numpy as np

def get_mri(path, ret_type:str='map'):
    '''
    Loads an MRI file.

    :param path: path to .nii / .nii.gz file
    :param ret_type: 'map' for a mapping of the mri file, or 'load' to load the mri directly to RAM
    '''
    if ret_type == 'map':
        return nib.load(path).dataobj
    elif ret_type == 'load':
        return nib.load(path).get_fdata().astype(np.float32)
    else:
        raise ValueError(f'ret_type must be \'map\' or \'load\', but given {ret_type}')

def get_gt_layer(mri, layer):
    mri_image = mri[:, :, layer]
    return mri_image.astype(bool)

def load_image_and_gt(folder):
    """
    Given a leaf folder, load MRI + ground truth segmentation objects.
    Prefers .npy if present, otherwise loads .nii.gz.
    Returns (mri_img, gt_img) or Error.
    """

    files = os.listdir(folder)

    # Use preprocessed npy files if they exist
    npy_mri = None
    npy_seg = None

    for f in files:
        if f.startswith("preprocessed") and "FLAIR" in f and f.endswith(".npy"):
            npy_mri = os.path.join(folder, f)
        if f.startswith("preprocessed") and "segmentation" in f and f.endswith(".npy"):
            npy_seg = os.path.join(folder, f)

    if npy_mri and npy_seg:
        mri_img = np.load(npy_mri)
        gt_img = np.load(npy_seg)
        return mri_img, gt_img

    # Fall back to .nii.gz
    nii_mri = None
    nii_seg = None

    for f in files:
        if f.endswith(".nii.gz") and "seg" in f:
            nii_seg = os.path.join(folder, f)
        elif f.endswith(".nii.gz") and ("t2f" in f or "flair" in f):
            nii_mri = os.path.join(folder, f)

    if nii_mri and nii_seg:
        mri_img = get_mri(nii_mri)
        gt_img = get_mri(nii_seg)
        return mri_img, gt_img
    raise RuntimeError(f"MRI files not found in folder {folder}")
    # return None, None

def find_instances(root):
    """
    Returns a list of tuples:
    (instance_name, leaf_folder_path)
    """
    instances = []

    for dirpath, dirnames, filenames in os.walk(root):
        if filenames and not dirnames:
            # Leaf folder found
            if os.path.basename(dirpath) == "FLAIR":
                instance_name = os.path.basename(os.path.dirname(dirpath))
            else:
                instance_name = os.path.basename(dirpath)

            instances.append((instance_name, dirpath))

    return instances
