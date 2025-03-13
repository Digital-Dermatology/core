from typing import Callable, Tuple

import torch

from ....src.models.utils import ModelType
from .model import masked_vit_base, masked_vit_large, masked_vit_small, masked_vit_tiny


def patch_images(images: torch.Tensor, patch_size: int):
    """
    Transforms images into patched images.

    imgs: (N, C, H, W)
    x: (N, #patches, patch_size**2 * C)
    """
    # make sure the image properties are correct
    img_is_square = images.shape[2] == images.shape[3]
    img_is_patchable = images.shape[2] % patch_size == 0
    assert img_is_square and img_is_patchable

    channels = images.shape[1]
    h = w = images.shape[2] // patch_size
    x = images.reshape(shape=(images.shape[0], channels, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(images.shape[0], h * w, patch_size**2 * channels))
    return x


def unpatch_images(x: torch.Tensor, patch_size: int):
    """
    Transforms patched images into images.

    x: (N, #patches, patch_size**2 * C)
    imgs: (N, C, H, W)
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, h * patch_size))
    return imgs


MASKED_VIT_DICT = {
    "masked_vit_tiny": masked_vit_tiny,
    "masked_vit_small": masked_vit_small,
    "masked_vit_base": masked_vit_base,
    "masked_vit_large": masked_vit_large,
}


def get_model_class(base_model_name: str) -> Tuple[Callable, ModelType]:
    encoder_cls = MASKED_VIT_DICT.get(base_model_name, None)
    model_type = ModelType.VIT
    if encoder_cls is None:
        raise ValueError(f"Invalid base model name: {base_model_name}")
    return encoder_cls, model_type
