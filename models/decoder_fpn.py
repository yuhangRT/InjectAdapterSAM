"""
Lightweight FPN decoder for WireCR-SAM semantic segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crnet_blocks import ConvBN

__all__ = ["WireCRFPNDecoder"]


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        ConvBN(in_channels, out_channels, kernel_size),
        nn.ReLU(inplace=True),
    )


class WireCRFPNDecoder(nn.Module):
    """FPN-like decoder over four 256-channel feature levels."""

    def __init__(self, in_channels: int = 256, fpn_channels: int = 128, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.fpn_channels = fpn_channels
        self.num_classes = num_classes

        self.lateral_c2 = ConvBN(in_channels, fpn_channels, 1)
        self.lateral_c3 = ConvBN(in_channels, fpn_channels, 1)
        self.lateral_c4 = ConvBN(in_channels, fpn_channels, 1)
        self.lateral_c5 = ConvBN(in_channels, fpn_channels, 1)

        self.smooth_p2 = _conv_bn_relu(fpn_channels, fpn_channels, 3)
        self.smooth_p3 = _conv_bn_relu(fpn_channels, fpn_channels, 3)
        self.smooth_p4 = _conv_bn_relu(fpn_channels, fpn_channels, 3)
        self.smooth_p5 = _conv_bn_relu(fpn_channels, fpn_channels, 3)

        self.fuse_conv = _conv_bn_relu(fpn_channels * 4, fpn_channels, 3)
        self.cls_head = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    @staticmethod
    def _upsample_add(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return target + upsampled

    def forward(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        c2 = features["c2"]
        c3 = features["c3"]
        c4 = features["c4"]
        c5 = features["c5"]

        p5 = self.lateral_c5(c5)
        p4 = self._upsample_add(p5, self.lateral_c4(c4))
        p3 = self._upsample_add(p4, self.lateral_c3(c3))
        p2 = self._upsample_add(p3, self.lateral_c2(c2))

        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)

        fused = torch.cat(
            [
                p2,
                F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(p4, size=p2.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(p5, size=p2.shape[-2:], mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        fused_feat = self.fuse_conv(fused)
        main_logits = self.cls_head(fused_feat)
        return main_logits, fused_feat
