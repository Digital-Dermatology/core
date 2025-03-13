import torch

from ..models.mae.utils import patch_images


class MAELoss(torch.nn.Module):
    def __init__(
        self,
        patch_size: int,
        norm_pix_loss: bool = False,
    ):
        super(MAELoss, self).__init__()
        self.patch_size = patch_size
        # (per-patch) normalized pixels as targets for computing loss
        self.norm_pix_loss = norm_pix_loss

    def forward(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        imgs: [B, C, H, W]
        pred: [B, #patches, p*p*C]
        mask: [B, #patches] -> 0 is keep, 1 is remove
        """
        target = patch_images(images=images, patch_size=self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (predictions - target) ** 2
        # [N, #patches], mean loss per patch
        loss = loss.mean(dim=-1)
        # mean loss on removed patches
        loss = (loss * masks).sum() / masks.sum()
        return loss
