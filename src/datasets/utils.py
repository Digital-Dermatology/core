from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder


def get_train_validation_data_loaders(
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    val_size: float,
    test_size: float = 0.0,
):
    # obtain training indices that will be used for validation
    num_train = len(ds)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    # test split if requested
    if test_size > 0.0:
        split_test = int(np.floor(test_size * num_train))
        train_idx, test_idx = indices[split_test:], indices[:split_test]
        split_val = int(np.floor(val_size * len(train_idx)))
        train_idx, valid_idx = train_idx[split_val:], train_idx[:split_val]
        train_loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=num_workers,
            drop_last=True,
            shuffle=False,
        )
        valid_loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(valid_idx),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
        test_loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
        return train_loader, valid_loader, test_loader
    else:
        split = int(np.floor(val_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=num_workers,
            drop_last=True,
            shuffle=False,
        )
        valid_loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(valid_idx),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
        return train_loader, valid_loader


def create_image_dataset(path: Union[str, list], transform: torch.nn.Module):
    # check if there are multiple paths or only single
    if type(path) is list:
        l_datasets = []
        for p in path:
            _dataset = ImageFolder(p, transform=transform)
            l_datasets.append(_dataset)
        dataset = ConcatDataset(l_datasets)
    elif type(path) is str:
        dataset = ImageFolder(path, transform=transform)
    else:
        raise ValueError(f"Unknown type of the training path: {type(path)}")
    return dataset


def clahe(image_name):
    image = cv2.imread(image_name, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    he_img = clahe.apply(image)
    image = cv2.cvtColor(he_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(image)
