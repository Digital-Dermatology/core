from typing import Union

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..augmentations.augmentations import GaussianBlur, Solarization


class MultiCropAugmentation(torch.nn.Module):
    def __init__(
        self,
        global_crops_scale: Union[tuple, str] = (0.32, 1.0),
        local_crops_scale: Union[tuple, str] = (0.05, 0.32),
        global_crops_number: int = 2,
        local_crops_number: int = 8,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
        remove_all_augmentation: bool = False,
        apply_random_rotation: bool = False,
        apply_random_distortion: bool = False,
        apply_aug_mix: bool = False,
        apply_random_invert: bool = False,
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

        # augmentations
        l_augmentations = []
        if not remove_all_augmentation:
            l_augmentations += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]

        # custom augmentations added to test behaviour of multi-crop
        if apply_random_rotation and not remove_all_augmentation:
            l_augmentations.append(
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(
                            degrees=(0, 180),
                        )
                    ],
                    p=0.5,
                )
            )
        if apply_random_distortion and not remove_all_augmentation:
            l_augmentations.append(
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
            )
        if apply_aug_mix and not remove_all_augmentation:
            l_augmentations.append(transforms.AugMix())
        if apply_random_invert and not remove_all_augmentation:
            l_augmentations.append(transforms.RandomInvert(p=0.2))
        l_augmentations = transforms.Compose(l_augmentations)

        # 1st global crop
        self.global_trans1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                l_augmentations,
                torch.nn.Identity() if remove_all_augmentation else GaussianBlur(p=1.0),
                normalize,
            ]
        )

        # 2nd global crop
        self.global_trans2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                l_augmentations,
                torch.nn.Identity() if remove_all_augmentation else GaussianBlur(p=0.1),
                torch.nn.Identity() if remove_all_augmentation else Solarization(p=0.2),
                normalize,
            ]
        )

        # transformation for the local small crops
        self.local_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                l_augmentations,
                torch.nn.Identity() if remove_all_augmentation else GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        # global crops
        crops.append(self.global_trans1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_trans2(image))
        # local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_trans(image))
        return crops


class DINODataAugmentation(MultiCropAugmentation):
    def __init__(
        self,
        global_crops_scale: Union[tuple, str] = (0.32, 1.0),
        local_crops_scale: Union[tuple, str] = (0.05, 0.32),
        local_crops_number: int = 8,
        **kwargs,
    ):
        # remove the arguments if given
        _ = kwargs.pop("global_crops_number", None)
        super().__init__(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            global_crops_number=2,
            apply_random_rotation=False,
            apply_random_distortion=False,
            apply_aug_mix=False,
            **kwargs,
        )


class iBOTDataAugmentation(MultiCropAugmentation):
    def __init__(
        self,
        global_crops_scale: Union[tuple, str] = (0.32, 1.0),
        local_crops_scale: Union[tuple, str] = (0.05, 0.32),
        global_crops_number: int = 2,
        local_crops_number: int = 8,
        **kwargs,
    ):
        super().__init__(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            global_crops_number=global_crops_number,
            apply_random_rotation=False,
            apply_random_distortion=False,
            apply_aug_mix=False,
            **kwargs,
        )
