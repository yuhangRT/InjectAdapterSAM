"""
Helpers for matching proposal boxes to ground-truth instances.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "box_iou",
    "greedy_match_boxes",
    "proposal_recall_at_k",
]


def _proposal_box_tensor(proposal: dict[str, Any]) -> torch.Tensor:
    for key in ("bbox", "bbox_processed", "bbox_full"):
        if key in proposal:
            return proposal[key]
    raise KeyError("Proposal is missing a bbox-like field. Expected one of: bbox, bbox_processed, bbox_full.")


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device if boxes1.numel() else boxes2.device)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def greedy_match_boxes(
    proposals: list[dict[str, Any]],
    target_instances: dict[str, Any],
    iou_threshold: float = 0.3,
) -> list[tuple[int, int]]:
    if not proposals:
        return []
    proposal_boxes = torch.stack([_proposal_box_tensor(proposal) for proposal in proposals], dim=0).float()
    proposal_labels = torch.tensor([proposal["category_id"] for proposal in proposals], dtype=torch.long, device=proposal_boxes.device)
    gt_boxes = target_instances["boxes"].to(proposal_boxes.device)
    gt_labels = target_instances["labels"].to(proposal_boxes.device)
    ious = box_iou(proposal_boxes, gt_boxes)
    matches = []
    used_gt = set()
    scored_pairs = []
    for proposal_idx in range(len(proposals)):
        for gt_idx in range(gt_boxes.shape[0]):
            if int(proposal_labels[proposal_idx].item()) != int(gt_labels[gt_idx].item()):
                continue
            score = float(ious[proposal_idx, gt_idx].item())
            if score >= iou_threshold:
                scored_pairs.append((score, proposal_idx, gt_idx))
    for _, proposal_idx, gt_idx in sorted(scored_pairs, reverse=True):
        if gt_idx in used_gt:
            continue
        matches.append((proposal_idx, gt_idx))
        used_gt.add(gt_idx)
    return matches


def proposal_recall_at_k(
    proposals: list[dict[str, Any]],
    target_instances: dict[str, Any],
    *,
    topk: int,
    iou_threshold: float = 0.5,
) -> float:
    gt_boxes = target_instances["boxes"]
    gt_labels = target_instances["labels"]
    if gt_boxes.numel() == 0:
        return 1.0
    proposals = sorted(proposals, key=lambda item: item["score"], reverse=True)[:topk]
    if not proposals:
        return 0.0
    matches = greedy_match_boxes(proposals, target_instances, iou_threshold=iou_threshold)
    matched_gt = {gt_idx for _, gt_idx in matches}
    return len(matched_gt) / max(int(gt_boxes.shape[0]), 1)
