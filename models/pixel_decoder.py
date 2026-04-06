"""Pixel decoder with stride-4 mask features for WireCR-HQInstSAM."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WireCRPixelDecoder"]


def _conv_gn_gelu(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.GroupNorm(32, out_channels),
        nn.GELU(),
    )


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = _conv_gn_gelu(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.refine(x)


class WireCRPixelDecoder(nn.Module):
    """Fuse WireCR-adapted features and emit stride-4 mask features."""

    feature_names = ("c2", "c3", "c4", "c5")

    def __init__(self, in_channels: int = 256, out_channels: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.laterals = nn.ModuleDict(
            {
                feature_name: _conv_gn_gelu(in_channels, out_channels, 1)
                for feature_name in self.feature_names
            }
        )
        self.merge_blocks = nn.ModuleDict(
            {
                "p4": _conv_gn_gelu(out_channels, out_channels, 3),
                "p3": _conv_gn_gelu(out_channels, out_channels, 3),
                "p2": _conv_gn_gelu(out_channels, out_channels, 3),
            }
        )
        self.upsample_stride8 = _UpsampleBlock(out_channels)
        self.upsample_stride4 = _UpsampleBlock(out_channels)

    def forward(self, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]]:
        missing = [feature_name for feature_name in self.feature_names if feature_name not in features]
        if missing:
            raise KeyError(f"Missing required multiscale features: {missing}")

        c2 = self.laterals["c2"](features["c2"])
        c3 = self.laterals["c3"](features["c3"])
        c4 = self.laterals["c4"](features["c4"])
        c5 = self.laterals["c5"](features["c5"])

        p5 = c5
        p4 = self.merge_blocks["p4"](c4 + p5)
        p3 = self.merge_blocks["p3"](c3 + p4)
        p2 = self.merge_blocks["p2"](c2 + p3)

        stride8_memory = self.upsample_stride8(p2)
        mask_features = self.upsample_stride4(stride8_memory)
        multi_scale_memory = (p2, p3, p4, p5)

        return {
            "mask_features": mask_features,
            "multi_scale_memory": multi_scale_memory,
            "stride8_memory": stride8_memory,
        }
