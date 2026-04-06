"""WireCR multi-scale adapter for instance-oriented SAM features."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn

from .crnet_adapter import WireCRAdapter, WireCRAdapterSimple
from .crnet_blocks import ConvBN

__all__ = ["WireCRMultiScaleAdapter"]


class WireCRMultiScaleAdapter(nn.Module):
    """Adapt SAM c2-c5 features with WireCR blocks and unify channels to 256."""

    feature_names = ("c2", "c3", "c4", "c5")

    def __init__(
        self,
        in_channels: Mapping[str, int] | int,
        out_channels: int = 256,
        c2_adapter_size: str = "small",
        c345_adapter_size: str = "medium",
        compression_ratio: int = 8,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = self._normalize_in_channels(in_channels)

        self.input_projections = nn.ModuleDict(
            {
                feature_name: nn.Sequential(
                    ConvBN(self.in_channels[feature_name], out_channels, 1),
                    nn.GELU(),
                )
                for feature_name in self.feature_names
            }
        )
        self.adapters = nn.ModuleDict(
            {
                "c2": WireCRAdapterSimple(
                    in_channels=out_channels,
                    adapter_size=c2_adapter_size,
                    use_residual=use_residual,
                ),
                "c3": WireCRAdapter(
                    in_channels=out_channels,
                    adapter_size=c345_adapter_size,
                    compression_ratio=compression_ratio,
                    use_residual=use_residual,
                ),
                "c4": WireCRAdapter(
                    in_channels=out_channels,
                    adapter_size=c345_adapter_size,
                    compression_ratio=compression_ratio,
                    use_residual=use_residual,
                ),
                "c5": WireCRAdapter(
                    in_channels=out_channels,
                    adapter_size=c345_adapter_size,
                    compression_ratio=compression_ratio,
                    use_residual=use_residual,
                ),
            }
        )

    def _normalize_in_channels(self, in_channels: Mapping[str, int] | int) -> Dict[str, int]:
        if isinstance(in_channels, int):
            return {feature_name: in_channels for feature_name in self.feature_names}

        missing = [feature_name for feature_name in self.feature_names if feature_name not in in_channels]
        if missing:
            raise ValueError(f"Missing in_channels entries for {missing}")
        return {feature_name: int(in_channels[feature_name]) for feature_name in self.feature_names}

    def forward(self, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        missing = [feature_name for feature_name in self.feature_names if feature_name not in features]
        if missing:
            raise KeyError(f"Missing required multiscale features: {missing}")

        adapted: Dict[str, torch.Tensor] = {}
        for feature_name in self.feature_names:
            projected = self.input_projections[feature_name](features[feature_name])
            adapted[feature_name] = self.adapters[feature_name](projected)
        return adapted
