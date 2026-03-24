"""
WireCR adapter modules for SAM feature enhancement.

This module redesigns CRNet's dual-path anisotropic convolutions into a
parameter-efficient residual adapter that operates on SAM image embeddings.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from .crnet_blocks import ConvBN, CRBlockGeneric, MultiScaleEncoder

__all__ = [
    "ADAPTER_CONFIGS",
    "WireCRAdapter",
    "WireCRAdapterSimple",
    "VanillaAdapter",
    "CRNetAdapter",
    "CRNetAdapterSimple",
    "crnet_adapter",
]


ADAPTER_CONFIGS = {
    "small": {
        "hidden_channels": 64,
        "encoder_channels": 32,
        "crblock_channels": 32,
    },
    "medium": {
        "hidden_channels": 128,
        "encoder_channels": 64,
        "crblock_channels": 64,
    },
    "large": {
        "hidden_channels": 256,
        "encoder_channels": 128,
        "crblock_channels": 128,
    },
}


class _AdapterBase(nn.Module):
    """Common utilities for WireCR adapters."""

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class WireCRAdapter(_AdapterBase):
    """
    Residual CRNet-inspired adapter for SAM embeddings.

    The adapter keeps CRNet's dual-path multi-scale design while using a
    stable compression-expansion bottleneck defined entirely in ``__init__``.
    """

    def __init__(
        self,
        in_channels=256,
        adapter_size="medium",
        compression_ratio=8,
        use_residual=True,
    ):
        super().__init__()

        if adapter_size not in ADAPTER_CONFIGS:
            raise ValueError(f"Invalid adapter_size: {adapter_size}")
        if compression_ratio not in [4, 8, 16, 32, 64]:
            raise ValueError(f"Invalid compression_ratio: {compression_ratio}")

        config = ADAPTER_CONFIGS[adapter_size]
        hidden_channels = config["hidden_channels"]
        encoder_channels = config["encoder_channels"]
        crblock_channels = config["crblock_channels"]
        compressed_channels = max(hidden_channels // compression_ratio, 4)

        self.in_channels = in_channels
        self.adapter_size = adapter_size
        self.compression_ratio = compression_ratio
        self.use_residual = use_residual

        self.input_proj = ConvBN(in_channels, hidden_channels, 1)
        self.encoder = MultiScaleEncoder(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            intermediate_channels=encoder_channels,
        )
        self.compression = nn.Sequential(
            OrderedDict(
                [
                    ("compress", ConvBN(hidden_channels, compressed_channels, 1)),
                    ("compress_relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("expand", ConvBN(compressed_channels, hidden_channels, 1)),
                    ("expand_relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ]
            )
        )
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            OrderedDict(
                [
                    ("conv5x5_bn", ConvBN(hidden_channels, hidden_channels, 5)),
                    ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("crblock1", CRBlockGeneric(hidden_channels, crblock_channels)),
                    ("crblock2", CRBlockGeneric(hidden_channels, crblock_channels)),
                ]
            )
        )
        self.output_proj = ConvBN(hidden_channels, in_channels, 1)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def forward(self, x):
        residual = x

        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.compression(x)
        x = x * (1.0 + self.context_gate(x))
        x = self.refine(x)
        x = self.output_proj(x) * self.output_scale

        if self.use_residual:
            x = x + residual

        return x


class WireCRAdapterSimple(_AdapterBase):
    """A simpler ablation variant without explicit compression-expansion."""

    def __init__(self, in_channels=256, adapter_size="medium", use_residual=True):
        super().__init__()

        if adapter_size not in ADAPTER_CONFIGS:
            raise ValueError(f"Invalid adapter_size: {adapter_size}")

        config = ADAPTER_CONFIGS[adapter_size]
        hidden_channels = config["hidden_channels"]
        encoder_channels = config["encoder_channels"]
        crblock_channels = config["crblock_channels"]

        self.use_residual = use_residual

        self.input_proj = ConvBN(in_channels, hidden_channels, 1)
        self.encoder = MultiScaleEncoder(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            intermediate_channels=encoder_channels,
        )
        self.refine = nn.Sequential(
            OrderedDict(
                [
                    ("crblock1", CRBlockGeneric(hidden_channels, crblock_channels)),
                    ("crblock2", CRBlockGeneric(hidden_channels, crblock_channels)),
                ]
            )
        )
        self.output_proj = ConvBN(hidden_channels, in_channels, 1)

        self._init_weights()

    def forward(self, x):
        residual = x

        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.refine(x)
        x = self.output_proj(x)

        if self.use_residual:
            x = x + residual

        return x


class VanillaAdapter(_AdapterBase):
    """A plain bottleneck convolutional adapter baseline without CR-style blocks."""

    def __init__(
        self,
        in_channels=256,
        adapter_size="medium",
        compression_ratio=8,
        use_residual=True,
    ):
        super().__init__()

        if adapter_size not in ADAPTER_CONFIGS:
            raise ValueError(f"Invalid adapter_size: {adapter_size}")
        if compression_ratio not in [4, 8, 16, 32, 64]:
            raise ValueError(f"Invalid compression_ratio: {compression_ratio}")

        config = ADAPTER_CONFIGS[adapter_size]
        hidden_channels = config["hidden_channels"]
        compressed_channels = max(hidden_channels // compression_ratio, 4)

        self.use_residual = use_residual

        self.input_proj = ConvBN(in_channels, hidden_channels, 1)
        self.body = nn.Sequential(
            OrderedDict(
                [
                    ("relu1", nn.GELU()),
                    ("compress", ConvBN(hidden_channels, compressed_channels, 1)),
                    ("relu2", nn.GELU()),
                    ("spatial", ConvBN(compressed_channels, compressed_channels, 3)),
                    ("relu3", nn.GELU()),
                    ("expand", ConvBN(compressed_channels, hidden_channels, 1)),
                    ("relu4", nn.GELU()),
                ]
            )
        )
        self.output_proj = ConvBN(hidden_channels, in_channels, 1)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def forward(self, x):
        residual = x

        x = self.input_proj(x)
        x = self.body(x)
        x = self.output_proj(x) * self.output_scale

        if self.use_residual:
            x = x + residual

        return x


CRNetAdapter = WireCRAdapter
CRNetAdapterSimple = WireCRAdapterSimple


def crnet_adapter(
    in_channels=256,
    adapter_size="medium",
    compression_ratio=8,
    use_residual=True,
    simple=False,
    adapter_kind="wirecr",
):
    """Factory function for WireCR adapter variants."""
    adapter_kind = str(adapter_kind).strip().lower()
    if adapter_kind not in {"wirecr", "vanilla"}:
        raise ValueError(f"Invalid adapter_kind: {adapter_kind}")

    if adapter_kind == "vanilla":
        return VanillaAdapter(
            in_channels=in_channels,
            adapter_size=adapter_size,
            compression_ratio=compression_ratio,
            use_residual=use_residual,
        )

    if simple:
        return WireCRAdapterSimple(
            in_channels=in_channels,
            adapter_size=adapter_size,
            use_residual=use_residual,
        )

    return WireCRAdapter(
        in_channels=in_channels,
        adapter_size=adapter_size,
        compression_ratio=compression_ratio,
        use_residual=use_residual,
    )
