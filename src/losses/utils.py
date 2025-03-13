import numpy as np
import segmentation_models_pytorch as smp
import torch


def get_segmentation_loss(loss_fn_name: str, mode: str):
    if loss_fn_name == "dice":
        criterion = smp.losses.DiceLoss(mode=mode)
    elif loss_fn_name == "tversky":
        criterion = smp.losses.TverskyLoss(mode=mode)
    elif loss_fn_name == "focal":
        criterion = smp.losses.FocalLoss(mode=mode)
    else:
        raise ValueError(f"Unrecognized loss function: {loss_fn_name}")
    return criterion


def mixup_data(x, y, alpha: float = 1.0, device: str = "cpu"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
