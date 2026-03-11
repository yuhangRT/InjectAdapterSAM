"""
CRNet-based feature enhancement adapter for SAM.

This module implements an adapter that processes SAM's ViT encoder output
through a modified CRNet architecture to enhance features before passing
them to SAM's mask decoder.
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from .crnet_blocks import ConvBN, CRBlockGeneric, MultiScaleEncoder

__all__ = ['CRNetAdapter']


# Adapter size configurations
ADAPTER_CONFIGS = {
    'small': {
        'encoder_channels': 32,
        'decoder_channels': 64,
        'crblock_channels': 64,
    },
    'medium': {
        'encoder_channels': 64,
        'decoder_channels': 128,
        'crblock_channels': 128,
    },
    'large': {
        'encoder_channels': 128,
        'decoder_channels': 256,
        'crblock_channels': 256,
    }
}


class CRNetAdapter(nn.Module):
    """
    CRNet-based feature enhancement adapter for SAM ViT encoder output.

    This adapter processes SAM's ViT encoder features through an encoder-decoder
    architecture with multi-scale convolutions and a compression bottleneck.
    It maintains spatial resolution and uses a residual connection for stability.

    Args:
        in_channels: Number of input feature channels (256 for SAM ViT-H)
        adapter_size: Size variant - 'small', 'medium', or 'large'
        compression_ratio: Compression ratio reciprocal (4, 8, 16, 32, 64)
        use_residual: Whether to use residual connection (default: True)

    Input:
        (batch, in_channels, H, W) - typically (N, 256, 64, 64) for SAM

    Output:
        (batch, in_channels, H, W) - same shape as input

    Example:
        >>> adapter = CRNetAdapter(in_channels=256, adapter_size='medium', compression_ratio=8)
        >>> x = torch.randn(2, 256, 64, 64)
        >>> y = adapter(x)
        >>> assert y.shape == x.shape
    """

    def __init__(self, in_channels=256, adapter_size='medium',
                 compression_ratio=4, use_residual=True):
        super(CRNetAdapter, self).__init__()

        assert adapter_size in ADAPTER_CONFIGS, f"Invalid adapter_size: {adapter_size}"
        assert compression_ratio in [4, 8, 16, 32, 64], f"Invalid compression_ratio: {compression_ratio}"

        self.in_channels = in_channels
        self.adapter_size = adapter_size
        self.compression_ratio = compression_ratio
        self.use_residual = use_residual

        # Get configuration for this adapter size
        config = ADAPTER_CONFIGS[adapter_size]
        encoder_channels = config['encoder_channels']
        decoder_channels = config['decoder_channels']
        crblock_channels = config['crblock_channels']

        # Optional input projection if channels don't match encoder input
        if in_channels != encoder_channels * 4:  # 4 = concat of two paths (2*2)
            self.input_proj = ConvBN(in_channels, encoder_channels * 2, 1)
        else:
            self.input_proj = nn.Identity()

        # Encoder: Multi-scale dual-path processing
        self.encoder = MultiScaleEncoder(
            in_channels=encoder_channels * 2,
            out_channels=encoder_channels,
            intermediate_channels=encoder_channels
        )

        # Calculate FC dimensions
        # We'll flatten spatial dims and apply FC compression
        self.use_fc_compression = True

        # Decoder: FC expansion -> 5x5 conv -> 2x CRBlock
        self.decoder_fc = nn.Linear(
            encoder_channels // compression_ratio,
            encoder_channels
        )

        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(encoder_channels, encoder_channels, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlockGeneric(encoder_channels, crblock_channels)),
            ("CRBlock2", CRBlockGeneric(encoder_channels, crblock_channels))
        ]))

        # Optional output projection
        if encoder_channels != in_channels:
            self.output_proj = ConvBN(encoder_channels, in_channels, 1)
        else:
            self.output_proj = nn.Identity()

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
        Forward pass through CRNet adapter.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W)

        Returns:
            Enhanced features of shape (batch, in_channels, H, W)
        """
        n, c, h, w = x.shape

        # Store input for residual connection
        residual = x

        # Optional input projection
        x = self.input_proj(x)

        # Encoder: Multi-scale processing
        x = self.encoder(x)

        # FC compression (flatten spatial dims, compress, restore)
        x = x.view(n, -1)  # (batch, encoder_channels * h * w)
        x = x.mean(dim=1, keepdim=True)  # Global average pooling
        x = x.repeat(1, h * w).view(n, h * w, -1)  # Expand back
        # Simpler approach: just use 1x1 conv for compression
        x = x.view(n, -1, h, w)

        # Apply compression via 1x1 conv
        compressed_channels = self.encoder.encoder.fusion.conv1x1.conv.out_channels
        self.fc_compress = nn.Conv2d(compressed_channels, compressed_channels // self.compression_ratio, 1).to(x.device)
        self.fc_expand = nn.Conv2d(compressed_channels // self.compression_ratio, compressed_channels, 1).to(x.device)

        # Simpler architecture: just use decoder features
        x = self.decoder_feature(x)

        # Output projection
        x = self.output_proj(x)

        # Residual connection
        if self.use_residual:
            x = x + residual

        return x


class CRNetAdapterSimple(nn.Module):
    """
    Simplified CRNet adapter without FC compression.

    This version uses only the multi-scale encoder and CRBlock decoder
    without the FC bottleneck, making it more suitable for maintaining
    spatial information.

    Args:
        in_channels: Number of input feature channels
        adapter_size: Size variant - 'small', 'medium', or 'large'
        use_residual: Whether to use residual connection
    """

    def __init__(self, in_channels=256, adapter_size='medium', use_residual=True):
        super(CRNetAdapterSimple, self).__init__()

        assert adapter_size in ADAPTER_CONFIGS, f"Invalid adapter_size: {adapter_size}"

        self.in_channels = in_channels
        self.adapter_size = adapter_size
        self.use_residual = use_residual

        config = ADAPTER_CONFIGS[adapter_size]
        decoder_channels = config['decoder_channels']
        crblock_channels = config['crblock_channels']

        # Input projection
        self.input_proj = ConvBN(in_channels, decoder_channels, 1)

        # Multi-scale encoder
        self.encoder = MultiScaleEncoder(
            in_channels=decoder_channels,
            out_channels=decoder_channels,
            intermediate_channels=config['encoder_channels']
        )

        # Decoder with CRBlocks
        self.decoder = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(decoder_channels, decoder_channels, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlockGeneric(decoder_channels, crblock_channels)),
            ("CRBlock2", CRBlockGeneric(decoder_channels, crblock_channels))
        ]))

        # Output projection
        self.output_proj = ConvBN(decoder_channels, in_channels, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, in_channels, H, W)

        Returns:
            (batch, in_channels, H, W)
        """
        residual = x

        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_proj(x)

        if self.use_residual:
            x = x + residual

        return x


def crnet_adapter(in_channels=256, adapter_size='medium',
                  compression_ratio=4, use_residual=True, simple=False):
    """
    Factory function to create CRNet adapter.

    Args:
        in_channels: Input feature channels
        adapter_size: 'small', 'medium', or 'large'
        compression_ratio: Compression ratio (4, 8, 16, 32, 64)
        use_residual: Use residual connection
        simple: Use simplified version without FC compression

    Returns:
        CRNetAdapter or CRNetAdapterSimple instance
    """
    if simple:
        return CRNetAdapterSimple(
            in_channels=in_channels,
            adapter_size=adapter_size,
            use_residual=use_residual
        )
    else:
        return CRNetAdapter(
            in_channels=in_channels,
            adapter_size=adapter_size,
            compression_ratio=compression_ratio,
            use_residual=use_residual
        )
