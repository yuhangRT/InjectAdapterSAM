"""
Loss functions for WireCR-InstSAM.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sam_metrics import compute_cldice

__all__ = [
    "CenterNetProposalLoss",
    "InstanceRefineLoss",
]


class CenterNetProposalLoss(nn.Module):
    def __init__(
        self,
        *,
        alpha: float = 2.0,
        beta: float = 4.0,
        hole_positive_weight: float = 1.5,
        size_weight: float = 1.0,
        offset_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.hole_positive_weight = float(hole_positive_weight)
        self.size_weight = float(size_weight)
        self.offset_weight = float(offset_weight)

    def forward(self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred_heatmap = torch.sigmoid(predictions["center_heatmap"]).clamp(1e-4, 1 - 1e-4)
        pred_size = predictions["size_map"]
        pred_offset = predictions["offset_map"]
        target_heatmap = targets["heatmap"].to(pred_heatmap.device)
        positive_mask = targets["positive_mask"].to(pred_heatmap.device)
        offset_mask = targets["offset_mask"].to(pred_heatmap.device)
        positive_weight = targets["positive_weight"].to(pred_heatmap.device)

        pos_inds = target_heatmap.eq(1).float()
        neg_inds = target_heatmap.lt(1).float()
        neg_weights = torch.pow(1 - target_heatmap, self.beta)
        pos_loss = -torch.log(pred_heatmap) * torch.pow(1 - pred_heatmap, self.alpha) * pos_inds
        pos_loss = pos_loss * positive_weight.unsqueeze(1)
        neg_loss = -torch.log(1 - pred_heatmap) * torch.pow(pred_heatmap, self.alpha) * neg_weights * neg_inds
        num_pos = pos_inds.sum().clamp(min=1.0)
        heatmap_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        size_target = targets["size_target"].to(pred_size.device)
        offset_target = targets["offset_target"].to(pred_offset.device)
        if positive_mask.any():
            size_loss = F.l1_loss(pred_size.permute(0, 2, 3, 1)[positive_mask], size_target.permute(0, 2, 3, 1)[positive_mask])
            offset_loss = F.l1_loss(
                pred_offset.permute(0, 2, 3, 1)[offset_mask],
                offset_target.permute(0, 2, 3, 1)[offset_mask],
            )
        else:
            size_loss = pred_size.sum() * 0.0
            offset_loss = pred_offset.sum() * 0.0
        total = heatmap_loss + self.size_weight * size_loss + self.offset_weight * offset_loss
        return {
            "loss": total,
            "heatmap_loss": heatmap_loss.detach(),
            "size_loss": size_loss.detach(),
            "offset_loss": offset_loss.detach(),
        }


class InstanceRefineLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1e-6) / (denom + 1e-6)
        return 1 - dice.mean()

    @staticmethod
    def _boundary_map(tensor: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(min=0.0, max=1.0)

    def forward(self, predictions: torch.Tensor, target_masks: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = []
        wire_losses = []
        hole_losses = []
        for pred_mask, target_mask, label in zip(predictions, target_masks, labels):
            pred_mask = pred_mask.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0).float()
            bce = F.binary_cross_entropy_with_logits(pred_mask, target_mask)
            dice = self._dice_loss(pred_mask, target_mask)
            if int(label.item()) == 1:
                boundary = F.l1_loss(self._boundary_map(torch.sigmoid(pred_mask)), self._boundary_map(target_mask))
                cldice = 1 - compute_cldice(torch.sigmoid(pred_mask), target_mask).mean()
                loss = bce + 0.5 * dice + 0.2 * boundary + 0.2 * cldice
                wire_losses.append(loss)
            else:
                loss = 1.5 * bce + 0.7 * dice
                hole_losses.append(loss)
            losses.append(loss)
        total_loss = torch.stack(losses).mean() if losses else predictions.sum() * 0.0
        return {
            "loss": total_loss,
            "wire_loss": torch.stack(wire_losses).mean().detach() if wire_losses else total_loss.detach() * 0.0,
            "hole_loss": torch.stack(hole_losses).mean().detach() if hole_losses else total_loss.detach() * 0.0,
        }
