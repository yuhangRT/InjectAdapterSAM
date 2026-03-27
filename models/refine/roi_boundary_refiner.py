"""
Optional lightweight ROI boundary refiner.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.crnet_blocks import ConvBN

__all__ = ["ROIBoundaryRefiner"]


class ROIBoundaryRefiner(nn.Module):
    def __init__(self, feature_channels: int = 256, hidden_channels: int = 64):
        super().__init__()
        self.feature_branch = nn.Sequential(
            ConvBN(feature_channels, hidden_channels, 3),
            nn.GELU(),
            ConvBN(hidden_channels, hidden_channels, 3),
            nn.GELU(),
        )
        self.mask_branch = nn.Sequential(
            ConvBN(1, hidden_channels, 3),
            nn.GELU(),
            ConvBN(hidden_channels, hidden_channels, 3),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            ConvBN(hidden_channels * 2, hidden_channels, 3),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, mask_logits: torch.Tensor, low_level_features: torch.Tensor) -> torch.Tensor:
        feature_embed = self.feature_branch(low_level_features)
        mask_embed = self.mask_branch(mask_logits)
        refined = self.refine(torch.cat([feature_embed, mask_embed], dim=1))
        return mask_logits + refined
