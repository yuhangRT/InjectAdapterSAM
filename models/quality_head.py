"""Quality head for refined instance masks."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["QualityHead"]


class QualityHead(nn.Module):
    """Predict refinement quality from features and coarse/decoder priors."""

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        refine_features: torch.Tensor,
        *,
        coarse_score: torch.Tensor,
        decoder_score: torch.Tensor,
    ) -> torch.Tensor:
        feature_vector = self.pool(refine_features)
        feature_vector = self.feature_mlp(feature_vector)
        score_vector = torch.stack((coarse_score.float(), decoder_score.float()), dim=-1)
        quality_logits = self.score_mlp(torch.cat((feature_vector, score_vector), dim=-1))
        return quality_logits.squeeze(-1).sigmoid()
