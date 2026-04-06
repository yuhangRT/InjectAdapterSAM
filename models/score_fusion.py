"""Explicit score fusion for final instance scoring."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ScoreFusion"]


class ScoreFusion(nn.Module):
    """Fuse class, box, coarse mask, and refine quality scores with an MLP."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        class_score: torch.Tensor,
        box_quality: torch.Tensor,
        coarse_mask_score: torch.Tensor,
        refine_quality_score: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack(
            (
                class_score.float(),
                box_quality.float(),
                coarse_mask_score.float(),
                refine_quality_score.float(),
            ),
            dim=-1,
        )
        return self.mlp(features).squeeze(-1).sigmoid()
