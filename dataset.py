import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from sklearn.model_selection import KFold
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def mri_normalize(mri):
    '''Normalizes an MRI entirely to value range [0, 1]'''
    mri = np.maximum(mri, 0)
    mri_min, mri_max = np.min(mri), np.max(mri)
    normalized = (mri - mri_min) / (mri_max - mri_min + 1e-8)
    return normalized

def get_mri(path):
    mri = nib.load(path).get_fdata().astype(np.float32)
    return mri_normalize(mri)

class Glioma3DDataset(Dataset):
    def __init__(self, root):
        """
        root: path to dataset root
        """
        self.instances = self._find_instances(root)

    def _find_instances(self, root):
        instances = []
        for dirpath, dirnames, filenames in os.walk(root):
            if filenames and not dirnames:
                instances.append(dirpath)
        return instances

    def _load_instance(self, folder):
        files = os.listdir(folder)

        # Prefer .npy
        mri = seg = None
        for f in files:
            if f.startswith("preprocessed") and "segmentation" in f and f.endswith(".npy"):
                seg = np.load(os.path.join(folder, f))
            elif f.startswith("preprocessed") and "FLAIR" in f and f.endswith(".npy"):
                mri = np.load(os.path.join(folder, f))

        if mri is not None and seg is not None:
            return mri, seg

        # Fall back to .nii.gz
        for f in files:
            if f.endswith(".nii.gz") and "seg" in f.lower():
                seg = get_mri(os.path.join(folder, f))
            elif f.endswith(".nii.gz") and ("t2f" in f or "flair" in f):
                mri = get_mri(os.path.join(folder, f))

        if mri is None or seg is None:
            raise RuntimeError(f"Missing MRI or GT in {folder}")

        return mri, seg

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        folder = self.instances[idx]
        mri, seg = self._load_instance(folder)

        # Add channel dim
        mri = torch.from_numpy(mri).unsqueeze(0).to(torch.float32)
        seg = torch.from_numpy(seg)
        seg = (seg > 0).float()

        return mri, seg


def build_cv_loaders(
    brats_training_root,
    # brats_validation_root,
    fold_idx,
    num_folds=5,
    batch_size=1,
    num_workers=4,
):
    full_train_ds = Glioma3DDataset(brats_training_root)
    # test_ds = Glioma3DDataset(brats_validation_root)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    splits = list(kf.split(range(len(full_train_ds))))

    train_indices, val_indices = splits[fold_idx]

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_train_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # test_loader = DataLoader(
    #     test_ds,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=True,
    # )

    return train_loader, val_loader#, test_loader

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

