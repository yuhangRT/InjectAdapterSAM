"""
Multi-feature SAM image encoder wrapper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["SAMMultiFeatureBackbone"]


def _default_out_indices(depth: int) -> tuple[int, int, int, int]:
    defaults = {
        12: (2, 5, 8, 11),
        24: (5, 11, 17, 23),
        32: (7, 15, 23, 31),
    }
    if depth not in defaults:
        raise ValueError(f"Unsupported SAM encoder depth for multi-feature extraction: {depth}")
    return defaults[depth]


class SAMMultiFeatureBackbone(nn.Module):
    """Extract intermediate SAM encoder features and normalize them to BCHW."""

    feature_names = ("c2", "c3", "c4", "c5")

    def __init__(self, image_encoder: nn.Module, out_indices: tuple[int, int, int, int] | None = None):
        super().__init__()
        self.image_encoder = image_encoder
        self.out_indices = tuple(out_indices or _default_out_indices(len(self.image_encoder.blocks)))
        self.embed_dim = self.image_encoder.patch_embed.proj.out_channels

        if len(self.out_indices) != 4:
            raise ValueError("SAMMultiFeatureBackbone requires exactly 4 out_indices.")

    def _to_bchw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 4 and tensor.dim() != 3:
            raise ValueError(f"Unsupported tensor rank for feature conversion: {tensor.shape}")

        if tensor.dim() == 4:
            if tensor.shape[-1] == self.embed_dim and tensor.shape[1] != self.embed_dim:
                return tensor.permute(0, 3, 1, 2).contiguous()
            return tensor.contiguous()

        batch_size, num_tokens, channels = tensor.shape
        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Cannot reshape token sequence of length {num_tokens} to square feature map.")
        return tensor.transpose(1, 2).reshape(batch_size, channels, side, side).contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        features = {}
        requested = set(self.out_indices)
        for block_index, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            if block_index in requested:
                feature_name = self.feature_names[self.out_indices.index(block_index)]
                features[feature_name] = self._to_bchw(x)

        if len(features) != 4:
            missing = [name for name in self.feature_names if name not in features]
            raise RuntimeError(f"Failed to capture all requested SAM features. Missing: {missing}")

        return features
