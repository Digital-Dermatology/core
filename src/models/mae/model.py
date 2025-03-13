from functools import partial
from typing import List

import torch
import torch.nn as nn

from ..encoders.vision_transformer import Block, VisionTransformer
from ..utils import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(VisionTransformer):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(
        self,
        img_size: List[int] = [224],
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        return_all_tokens: bool = False,
        masked_im_modeling: bool = False,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            return_all_tokens=return_all_tokens,
            masked_im_modeling=masked_im_modeling,
        )
        num_patches = self.patch_embed.num_patches
        # fixed sin-cos embedding
        self.pos_embed.requires_grad = False

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            data=torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        # decoder to patch
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size**2 * in_channels,
            bias=True,
        )
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
        return_helpers: bool = False,
        **kwargs,
    ):
        # embed patches
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # Perform per-sample random masking by per-sample shuffling.
        # Per-sample shuffling is done by argsort random noise.
        N, N_PATCHES, D = x.shape
        len_keep = int(N_PATCHES * (1 - mask_ratio))

        # generate random noise for the selection (in [0, 1])
        noise = torch.rand(N, N_PATCHES, device=x.device)
        # sort noise for each sample (ascend: small is keep, large is remove)
        ids_shuffle = torch.argsort(noise, dim=1)
        # IDs to restore the mask (used for the decoder)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset (random ids to keep from every sample)
        ids_keep = ids_shuffle[:, :len_keep]
        # reshape for alignment with "x"
        ids_keep = ids_keep.unsqueeze(-1).repeat(1, 1, D)
        # masked input
        x_masked = torch.gather(x, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, N_PATCHES], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        # binary mask used to create "x_masked"
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        if return_helpers:
            return x_masked, mask, ids_restore
        else:
            return x_masked

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        return_all_tokens=None,
        **kwargs,
    ):
        x, mask, ids_restore = self.prepare_tokens(
            x=imgs,
            mask_ratio=mask_ratio,
            return_helpers=True,
        )
        for blk in self.blocks:
            x = blk(x)
        latent = self.norm(x)
        # [N, N_PATCHES, p*p*3]
        pred = self.forward_decoder(x=latent, ids_restore=ids_restore)
        return_all_tokens = (
            self.return_all_tokens if return_all_tokens is None else return_all_tokens
        )
        if return_all_tokens:
            return imgs, pred, mask, latent
        return imgs, pred, mask, latent[:, 0]


def masked_vit_tiny(patch_size: int = 16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def masked_vit_small(patch_size: int = 16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def masked_vit_base(patch_size: int = 16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def masked_vit_large(patch_size: int = 16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model
