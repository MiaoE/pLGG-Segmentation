import os

# Fix for OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader, ConcatDataset

from pytorch3dunet.unet3d.utils import (
    get_logger,
    get_class
)

from pytorch3dunet.unet3d.config import TorchDevice, os_dependent_dataloader_kwargs
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("Dataset")

def get_dataset_loader(config: dict) -> dict[str, DataLoader]:
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).
    Args:
        config:  a top level configuration object containing the 'loaders' key
    Returns:
        dict {
            'train': <train_loader>
            'val': <val_loader>
        }
    """
    assert "loaders" in config, "Could not find data loaders configuration"
    loaders_config = config["loaders"]
    assert set(loaders_config["train"]["file_paths"]).isdisjoint(loaders_config["val"]["file_paths"]), (
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"
    )

    logger.info("Creating training and validation set loaders...")

    # get dataset class
    dataset_cls_str = loaders_config.get("dataset", None)
    if dataset_cls_str is None:
        dataset_cls_str = "StandardHDF5Dataset"
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    def _loader_classes(class_name):
        modules = ["pytorch3dunet.datasets.hdf5", "pytorch3dunet.datasets.dsb", "pytorch3dunet.datasets.utils"]
        return get_class(class_name, modules)
    dataset_class = _loader_classes(dataset_cls_str)

    train_datasets = dataset_class.create_datasets(loaders_config, phase="train")
    val_datasets = dataset_class.create_datasets(loaders_config, phase="val")

    num_workers = loaders_config.get("num_workers", 1)
    logger.info(f"Number of workers for train/val dataloader: {num_workers}")
    batch_size = loaders_config.get("batch_size", 1)
    device = config.get("device", None)
    assert device, "Device not specified in the config file and could not be inferred automatically"
    if device == TorchDevice.CUDA and torch.cuda.device_count() > 1:
        logger.info(
            f"{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}"
        )
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f"Batch size for train/val loader: {batch_size}")
    loader_kwargs = os_dependent_dataloader_kwargs()
    return {
        "train": DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            **loader_kwargs,
        ),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        "val": DataLoader(
            ConcatDataset(val_datasets),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **loader_kwargs,
        ),
    }