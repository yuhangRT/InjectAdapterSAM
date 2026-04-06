"""Refine-stage losses for WireCR-HQInstSAM."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RefineLossCriterion"]


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def _balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_pos_weight: float = 256.0,
) -> torch.Tensor:
    positives = targets.sum()
    negatives = targets.numel() - positives
    pos_weight = negatives / positives.clamp(min=1.0)
    pos_weight = pos_weight.clamp(min=1.0, max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=logits.new_tensor(float(pos_weight.item())),
    )


def _boundary_map(tensor: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def _mask_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.sigmoid() > 0.5
    targets = targets > 0.5
    intersection = (preds & targets).float().sum(dim=(1, 2, 3))
    union = (preds | targets).float().sum(dim=(1, 2, 3)).clamp(min=1.0)
    return intersection / union


class RefineLossCriterion(nn.Module):
    """Compute refine BCE, Dice, Boundary, and quality regression losses."""

    def __init__(
        self,
        *,
        loss_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.loss_weights = {
            "bce": 1.0,
            "dice": 1.0,
            "boundary": 0.5,
            "quality": 0.5,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

    def forward(
        self,
        *,
        refined_mask_logits: torch.Tensor,
        target_masks: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1)
        target_masks = target_masks.float().to(refined_mask_logits.device)

        if refined_mask_logits.numel() == 0:
            zero = refined_mask_logits.sum() * 0.0
            return {
                "loss_bce": zero,
                "loss_dice": zero,
                "loss_boundary": zero,
                "loss_quality": zero,
                "loss_total": zero,
                "loss": zero,
            }

        loss_bce = _balanced_bce_with_logits(refined_mask_logits, target_masks)
        loss_dice = _dice_loss(refined_mask_logits, target_masks).mean()
        pred_boundary = _boundary_map(refined_mask_logits.sigmoid())
        target_boundary = _boundary_map(target_masks)
        loss_boundary = F.l1_loss(pred_boundary, target_boundary)

        quality_target = _mask_iou(refined_mask_logits, target_masks)
        loss_quality = F.mse_loss(quality_scores.float(), quality_target)

        loss_total = (
            self.loss_weights["bce"] * loss_bce
            + self.loss_weights["dice"] * loss_dice
            + self.loss_weights["boundary"] * loss_boundary
            + self.loss_weights["quality"] * loss_quality
        )
        return {
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_boundary": loss_boundary,
            "loss_quality": loss_quality,
            "loss_total": loss_total,
            "loss": loss_total,
        }
