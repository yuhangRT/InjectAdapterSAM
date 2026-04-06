"""Class-wise mask NMS for final instance outputs."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

__all__ = ["ClassWiseMaskNMS", "mask_iou", "class_wise_mask_nms"]


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    intersection = (mask_a & mask_b).float().sum()
    union = (mask_a | mask_b).float().sum().clamp(min=1.0)
    return intersection / union


def class_wise_mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    iou_threshold: float = 0.6,
) -> List[int]:
    if masks.numel() == 0:
        return []

    order = torch.argsort(scores.float(), descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = []
        for candidate in order[1:]:
            candidate_index = int(candidate.item())
            same_class = int(labels[candidate_index].item()) == int(labels[current].item())
            if not same_class:
                remaining.append(candidate_index)
                continue
            overlap = float(mask_iou(masks[current], masks[candidate_index]).item())
            if overlap <= iou_threshold:
                remaining.append(candidate_index)
        order = scores.new_tensor(remaining, dtype=torch.long)
    return keep


class ClassWiseMaskNMS(nn.Module):
    """Module wrapper for class-wise mask NMS."""

    def __init__(self, iou_threshold: float = 0.6) -> None:
        super().__init__()
        self.iou_threshold = float(iou_threshold)

    def forward(self, masks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> List[int]:
        return class_wise_mask_nms(
            masks,
            scores,
            labels,
            iou_threshold=self.iou_threshold,
        )
