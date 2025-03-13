from typing import Union

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..augmentations.augmentations import RandomApply, Solarization


class SimCLRDataAugmentation(torch.nn.Module):
    def __init__(
        self,
        target_size=96,
        gaussian_kernel: int = 23,
        scaling: float = 1.0,
        two_augmentations=False,
    ):
        # configs
        self.target_shape = target_size
        self.two_augmentations = two_augmentations

        # normalization
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # data augmentation
        color_jitter = transforms.ColorJitter(
            0.8 * scaling,
            0.8 * scaling,
            0.8 * scaling,
            0.2 * scaling,
        )
        self.data_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.target_shape),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=gaussian_kernel, sigma=(0.1, 2)),
                normalize,
            ]
        )

        if self.two_augmentations:
            # in BYOL they used two different augmentations
            # the second one had slightly other probabilities
            # and includes solarization
            gauss = transforms.GaussianBlur(kernel_size=gaussian_kernel, sigma=(0.1, 2))
            self.data_aug_2 = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=self.target_shape),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    RandomApply(gauss, p=0.1),
                    Solarization(p=0.2),
                    normalize,
                ]
            )

    def __call__(self, image):
        # create two augmentations of the same image
        img_aug_1 = self.data_aug(image)
        if self.two_augmentations:
            img_aug_2 = self.data_aug_2(image)
        else:
            img_aug_2 = self.data_aug(image)
        return [img_aug_1, img_aug_2]


class MultiCropSimCLRAugmentation(torch.nn.Module):
    def __init__(
        self,
        global_crops_scale: Union[tuple, str] = (0.32, 1.0),
        local_crops_scale: Union[tuple, str] = (0.05, 0.32),
        global_crops_number: int = 2,
        local_crops_number: int = 8,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
        gaussian_kernel: int = 23,
        scaling: float = 1.0,
        remove_all_augmentation: bool = False,
    ):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        # evaluate if strings (caused by yaml file)
        if type(global_crops_scale) is str:
            global_crops_scale = eval(global_crops_scale)
        if type(local_crops_scale) is str:
            local_crops_scale = eval(local_crops_scale)

        # normalization
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        color_jitter = transforms.ColorJitter(
            0.8 * scaling,
            0.8 * scaling,
            0.8 * scaling,
            0.2 * scaling,
        )
        self.data_aug = transforms.Compose(
            [
                normalize,
            ]
        )

        # augmentations
        l_augmentations = []
        if not remove_all_augmentation:
            l_augmentations += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=gaussian_kernel, sigma=(0.1, 2)),
            ]
        l_augmentations = transforms.Compose(l_augmentations)

        self.global_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                l_augmentations,
                normalize,
            ]
        )
        self.local_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                l_augmentations,
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        # global crops
        crops.append(self.global_trans(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_trans(image))
        # local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_trans(image))
        return crops
