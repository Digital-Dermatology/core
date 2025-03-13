from torch import nn

from ..encoders.utils import get_encoder_class
from ..utils import ModelType
from .predictor import MLP


class BYOLModel(nn.Module):
    def __init__(
        self,
        base_model: str,
        projection_size=256,
        projection_hidden_size=4096,
        **kwargs,
    ):
        super(BYOLModel, self).__init__()
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
        # projection head
        self.projection = MLP(
            in_channels=n_feat,
            projection_size=projection_size,
            hidden_size=projection_hidden_size,
        )

    def forward(self, x, return_embedding=False):
        # embedding
        e = self.backbone(x)
        e = e.squeeze()
        if return_embedding:
            return e
        # project
        z = self.projection(e)
        return z
