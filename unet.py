import os
import random

# Fix for OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader, ConcatDataset

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import (
    TensorboardFormatter,
    create_lr_scheduler,
    create_optimizer,
    get_logger,
    get_number_of_learnable_parameters,
    get_class
)

from pytorch3dunet.unet3d.config import copy_config, load_config
from pytorch3dunet.unet3d.config import TorchDevice, os_dependent_dataloader_kwargs
from pytorch3dunet.unet3d.trainer import create_trainer, UNetTrainer
from pytorch3dunet.unet3d.utils import get_logger

from dataset import get_dataset_loader

logger = get_logger("UNet3DTraining")



def get_trainer(config: dict) -> "UNetTrainer":
    # Create the model
    model = get_model(config["model"])

    device = config.get("device", None)
    assert device, "Device not specified in the config file and could not be inferred automatically"
    logger.info(f"Using device: {device}")
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f"Number of learnable params {get_number_of_learnable_parameters(model)}")

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_dataset_loader(config)

    # Create the optimizer
    optimizer = create_optimizer(config["optimizer"], model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get("lr_scheduler", None), optimizer)

    trainer_config = config["trainer"]
    # Create tensorboard formatter
    tensorboard_formatter_config = trainer_config.pop("tensorboard_formatter", {})
    tensorboard_formatter = TensorboardFormatter(**tensorboard_formatter_config)
    # Create trainer
    resume = trainer_config.pop("resume", None)
    pre_trained = trainer_config.pop("pre_trained", None)

    return UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        loaders=loaders,
        tensorboard_formatter=tensorboard_formatter,
        resume=resume,
        pre_trained=pre_trained,
        device=device,
        **trainer_config,
    )

def main():
    """Main entry point for training 3D U-Net models.

    Loads configuration from command line arguments, sets random seeds if specified,
    creates a trainer instance, and starts the training process.
    """
    # Load and log experiment configuration
    config, config_path = load_config()
    logger.info(config)

    manual_seed = config.get("manual_seed", None)
    if manual_seed is not None:
        logger.info(f"Seed the RNG for all devices with {manual_seed}")
        logger.warning("Using CuDNN deterministic setting. This may slow down the training!")
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = get_trainer(config)
    # Copy config file
    copy_config(config, config_path)
    # Start training
    trainer.fit()


if __name__ == "__main__":
    main()