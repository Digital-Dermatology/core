from typing import Callable, Tuple

from torchvision import models as torchvision_models

from ....src.models.encoders.swin_transformer import swin_base, swin_small, swin_tiny
from ....src.models.encoders.vision_transformer import (
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)
from ....src.models.utils import ModelType

VIT_DICT = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "swin_tiny": swin_tiny,
    "swin_small": swin_small,
    "swin_base": swin_base,
}


def get_encoder_class(base_model_name: str) -> Tuple[Callable, ModelType]:
    encoder_cls = VIT_DICT.get(base_model_name, None)
    model_type = ModelType.VIT
    if encoder_cls is None:
        if base_model_name in torchvision_models.__dict__.keys():
            encoder_cls = torchvision_models.__dict__[base_model_name]
            model_type = ModelType.CNN
        else:
            raise ValueError(f"Invalid base model name: {base_model_name}")
    return encoder_cls, model_type
