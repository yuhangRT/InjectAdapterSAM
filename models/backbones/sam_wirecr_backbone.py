"""
SAM + WireCR multi-level backbone for instance segmentation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.crnet_adapter import crnet_adapter
from models.crnet_blocks import ConvBN

__all__ = ["SAMWireCRBackbone"]


class SAMWireCRBackbone(nn.Module):
    FEATURE_LEVELS = ("c2", "c3", "c4", "c5")

    def __init__(
        self,
        backend,
        *,
        out_channels: int = 256,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.image_encoder = backend.image_encoder
        self.out_channels = int(out_channels)
        self.freeze_encoder = bool(freeze_encoder)

        depth = len(self.image_encoder.blocks)
        self.out_indices = {
            12: (2, 5, 8, 11),
            24: (5, 11, 17, 23),
            32: (7, 15, 23, 31),
        }.get(depth)
        if self.out_indices is None:
            raise ValueError(f"Unsupported SAM encoder depth for feature extraction: {depth}")

        embed_dim = backend.embed_dim
        self.proj_c2 = ConvBN(embed_dim, out_channels, 1)
        self.proj_c3 = ConvBN(embed_dim, out_channels, 1)
        self.proj_c4 = ConvBN(embed_dim, out_channels, 1)
        self.proj_c5 = nn.Identity()

        self.adapter_c2 = nn.Identity()
        self.adapter_c3 = crnet_adapter(
            in_channels=out_channels,
            adapter_kind="wirecr",
            adapter_size="small",
            compression_ratio=16,
            use_residual=True,
            simple=True,
        )
        self.adapter_c4 = crnet_adapter(
            in_channels=out_channels,
            adapter_kind="wirecr",
            adapter_size="medium",
            compression_ratio=8,
            use_residual=True,
            simple=False,
        )
        self.adapter_c5 = crnet_adapter(
            in_channels=out_channels,
            adapter_kind="wirecr",
            adapter_size="medium",
            compression_ratio=8,
            use_residual=True,
            simple=False,
        )

        if self.freeze_encoder:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

    def _tokens_to_bchw(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 4:
            return tokens.permute(0, 3, 1, 2).contiguous()
        if tokens.dim() == 3:
            batch_size, num_tokens, channels = tokens.shape
            side = int(math.isqrt(num_tokens))
            return tokens.transpose(1, 2).reshape(batch_size, channels, side, side).contiguous()
        raise ValueError(f"Unsupported token tensor shape: {tokens.shape}")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.image_encoder.patch_embed(images)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        raw_features: dict[str, torch.Tensor] = {}
        for block_index, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            if block_index in self.out_indices:
                level_name = self.FEATURE_LEVELS[self.out_indices.index(block_index)]
                raw_features[level_name] = self._tokens_to_bchw(x)

        image_embeddings = self.image_encoder.neck(self._tokens_to_bchw(x))
        projected = {
            "c2": self.adapter_c2(self.proj_c2(raw_features["c2"])),
            "c3": self.adapter_c3(self.proj_c3(raw_features["c3"])),
            "c4": self.adapter_c4(self.proj_c4(raw_features["c4"])),
            "c5": self.adapter_c5(self.proj_c5(image_embeddings)),
            "image_embeddings": image_embeddings,
        }
        return projected

    def get_adapter_params(self) -> list[nn.Parameter]:
        modules = [
            self.proj_c2,
            self.proj_c3,
            self.proj_c4,
            self.adapter_c3,
            self.adapter_c4,
            self.adapter_c5,
        ]
        params = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params
