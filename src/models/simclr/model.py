import torch.nn as nn
import torch.nn.functional as F

from ..encoders.utils import get_encoder_class
from ..utils import ModelType


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model: str, out_dim: int, **kwargs):
        super(ResNetSimCLR, self).__init__()
        encoder_cls, model_type = get_encoder_class(base_model)
        if model_type is ModelType.VIT:
            self.backbone = encoder_cls(**kwargs)
            n_feat = self.backbone.embed_dim
        elif model_type is ModelType.CNN:
            encoder = encoder_cls(**kwargs)
            n_feat = encoder.fc.in_features
            self.backbone = nn.Sequential(*list(encoder.children())[:-1])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        # projection MLP
        self.dense1 = nn.Linear(n_feat, n_feat)
        self.dense2 = nn.Linear(n_feat, out_dim)

    def forward(self, z):
        # embed
        e = self.backbone(z)
        e = e.squeeze()
        # project
        z = self.dense1(e)
        z = F.relu(z)
        z = self.dense2(z)
        return e, z
