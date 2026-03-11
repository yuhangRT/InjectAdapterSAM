"""
Reusable CRNet building blocks for adapter architecture.

This module extracts and generalizes the core components from CRNet
to create flexible adapter modules for vision transformers.
"""

import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ['ConvBN', 'CRBlockGeneric', 'ResidualAdapter']


class ConvBN(nn.Sequential):
    """
    Convolutional layer with BatchNorm2d.

    Automatically calculates padding based on kernel size to maintain
    spatial dimensions (stride=1).

    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        kernel_size: Convolution kernel size (int or tuple)
        stride: Stride of the convolution
        groups: Number of groups for grouped convolution
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlockGeneric(nn.Module):
    """
    Generalized CRBlock for arbitrary input channels.

    This block maintains the dual-path multi-scale architecture from CRNet
    while allowing configurable channel dimensions. It captures features
    at multiple scales using different kernel sizes.

    Architecture:
        - Path 1 (long-range): 3×3 → 1×9 → 9×1 convolutions
        - Path 2 (short-range): 1×5 → 5×1 convolutions
        - Fusion: Concatenate → 1×1 conv → Residual connection

    Args:
        in_channels: Number of input channels
        intermediate_channels: Number of intermediate channels in dual paths
        leaky_slope: Negative slope for LeakyReLU (default: 0.3)
    """

    def __init__(self, in_channels, intermediate_channels=7, leaky_slope=0.3):
        super(CRBlockGeneric, self).__init__()

        # Path 1: Long-range feature extraction
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(in_channels, intermediate_channels, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)),
            ('conv1x9', ConvBN(intermediate_channels, intermediate_channels, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)),
            ('conv9x1', ConvBN(intermediate_channels, intermediate_channels, [9, 1])),
        ]))

        # Path 2: Short-range feature extraction
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(in_channels, intermediate_channels, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)),
            ('conv5x1', ConvBN(intermediate_channels, intermediate_channels, [5, 1])),
        ]))

        # Feature fusion: concatenate both paths and reduce
        self.conv1x1 = ConvBN(intermediate_channels * 2, in_channels, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through CRBlockGeneric.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W)

        Returns:
            Output tensor of shape (batch, in_channels, H, W)
        """
        identity = self.identity(x)

        # Dual-path processing
        out1 = self.path1(x)
        out2 = self.path2(x)

        # Feature fusion
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        # Residual connection
        out = self.relu(out + identity)
        return out


class ResidualAdapter(nn.Module):
    """
    Wrapper that adds residual connection to any adapter module.

    This allows the adapter to learn modifications while preserving
    the original input via residual connection: output = x + adapter(x).

    Args:
        adapter_module: The adapter module to wrap
    """

    def __init__(self, adapter_module):
        super(ResidualAdapter, self).__init__()
        self.adapter = adapter_module

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            x + adapter(x)
        """
        return x + self.adapter(x)


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder using dual-path architecture.

    This encoder processes features through two parallel paths with
    different receptive fields, then fuses them through 1x1 convolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        intermediate_channels: Number of channels in dual paths
    """

    def __init__(self, in_channels, out_channels, intermediate_channels):
        super(MultiScaleEncoder, self).__init__()

        # Path 1: Multi-scale long-range
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(in_channels, intermediate_channels, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(intermediate_channels, intermediate_channels, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(intermediate_channels, intermediate_channels, [9, 1])),
        ]))

        # Path 2: Short-range
        self.path2 = ConvBN(in_channels, intermediate_channels, 3)

        # Fusion
        self.fusion = nn.Sequential(OrderedDict([
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x1', ConvBN(intermediate_channels * 2, out_channels, 1)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through multi-scale encoder."""
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.fusion(out)
        return out
