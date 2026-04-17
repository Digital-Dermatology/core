from typing import Callable

import numpy as np
import torch

from .lars import LARS


def get_lin_scaled_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    lr: float,
    bs: int,
):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=lr * bs / 256,
            momentum=0.9,
            weight_decay=0,
        )
    elif optimizer_name == "sgd_wo_momentum":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=lr * bs / 256,
            momentum=0,
            weight_decay=0,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr * bs / 256,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=lr * bs / 256,
        )
    else:
        raise ValueError(f"Unrecognized optimizer name: {optimizer_name}")
    return optimizer


def get_optimizer_type(optimizer_name: str) -> Callable:
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "lars": LARS,
    }
    optimizer_cls = optimizer_dict.get(optimizer_name, np.nan)
    if optimizer_cls is np.nan:
        raise ValueError("Invalid optimizer name.")
    return optimizer_cls


def get_learning_rate_scheduler(
    scheduler_name,
    optimizer,
    total_steps: int,
    n_batches_per_epoch: int,
    scheduler_epoch_cycles: int,
    **scheduler_kwargs,
):
    from deepspeed.runtime.lr_schedules import WarmupDecayLR, WarmupLR

    if scheduler_name is None:
        scheduler = None
    elif scheduler_name == "WarmupDecayLR":
        scheduler = WarmupDecayLR(
            optimizer,
            total_num_steps=total_steps,
            **scheduler_kwargs,
        )
    elif scheduler_name == "WarmupLR":
        scheduler = WarmupLR(
            optimizer,
            **scheduler_kwargs,
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=scheduler_epoch_cycles * n_batches_per_epoch,
        )
    else:
        raise ValueError(f"Unknown learning rate scheduler: {scheduler_name}")
    return scheduler
