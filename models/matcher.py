"""Hungarian-style matcher for coarse instance predictions."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "HungarianMatcher",
    "box_iou",
    "generalized_box_iou",
    "linear_sum_assignment",
    "normalize_xyxy_boxes",
    "resolve_target_spatial_size",
]


def normalize_xyxy_boxes(boxes: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
    """Normalize xyxy boxes to [0, 1] using the provided (height, width)."""

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    height, width = spatial_size
    boxes = boxes.float()
    if float(boxes.detach().amax().item()) <= 1.0:
        return boxes.clamp(0.0, 1.0)

    scale = boxes.new_tensor([width, height, width, height])
    return (boxes / scale).clamp(0.0, 1.0)


def resolve_target_spatial_size(
    target: dict[str, torch.Tensor | Tuple[int, int] | Sequence[int]],
    fallback: Tuple[int, int],
) -> Tuple[int, int]:
    """Resolve the target image size used for absolute xyxy boxes."""

    for key in ("processed_size", "image_size", "input_size"):
        if key not in target:
            continue
        size = target[key]
        if torch.is_tensor(size):
            size = size.tolist()
        if len(size) != 2:
            raise ValueError(f"Expected {key} to provide (height, width), got {size!r}")
        return int(size[0]), int(size[1])
    return int(fallback[0]), int(fallback[1])


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    return widths * heights


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    iou = box_iou(boxes1, boxes2)
    top_left = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    area = wh[..., 0] * wh[..., 1]

    inter_top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - inter
    return iou - (area - union) / area.clamp(min=1e-6)


def _pairwise_bce_cost(pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    query_count, height, width = pred_masks.shape
    target_count = target_masks.shape[0]
    pred = pred_masks.unsqueeze(1).expand(query_count, target_count, height, width)
    target = target_masks.unsqueeze(0).expand(query_count, target_count, height, width)
    return F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean(dim=(2, 3))


def _pairwise_dice_cost(pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    pred_probs = pred_masks.sigmoid()
    intersection = (pred_probs.unsqueeze(1) * target_masks.unsqueeze(0)).sum(dim=(2, 3))
    denominator = pred_probs.unsqueeze(1).sum(dim=(2, 3)) + target_masks.unsqueeze(0).sum(dim=(2, 3))
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def linear_sum_assignment(cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python Hungarian solver for small to medium rectangular cost matrices."""

    if cost_matrix.dim() != 2:
        raise ValueError(f"Expected 2D cost matrix, got {tuple(cost_matrix.shape)}")

    rows, cols = cost_matrix.shape
    if rows == 0 or cols == 0:
        empty = torch.zeros((0,), dtype=torch.int64)
        return empty, empty

    transposed = False
    matrix = cost_matrix.detach().cpu()
    if rows > cols:
        matrix = matrix.t().contiguous()
        rows, cols = matrix.shape
        transposed = True

    inf = float("inf")
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = float(matrix[i0 - 1, j - 1].item()) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * rows
    for j in range(1, cols + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    row_ind = torch.arange(rows, dtype=torch.int64)
    col_ind = torch.tensor(assignment, dtype=torch.int64)
    valid = col_ind >= 0
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]
    if transposed:
        return col_ind, row_ind
    return row_ind, col_ind


class HungarianMatcher(nn.Module):
    """Match coarse predictions to targets with Hungarian-style assignment."""

    def __init__(
        self,
        *,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_mask_bce: float = 5.0,
        cost_mask_dice: float = 5.0,
    ) -> None:
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_mask_bce = float(cost_mask_bce)
        self.cost_mask_dice = float(cost_mask_dice)

    def _resize_target_masks(self, target_masks: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        if target_masks.numel() == 0:
            return target_masks.reshape(0, *spatial_size)
        resized = F.interpolate(
            target_masks.unsqueeze(1).float(),
            size=spatial_size,
            mode="nearest",
        )
        return resized[:, 0]

    def _compute_cost_matrix(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_masks: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> torch.Tensor:
        class_probs = pred_logits.softmax(dim=-1)
        cost_class = -class_probs[:, target_labels]
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
        cost_giou = -generalized_box_iou(pred_boxes, target_boxes)
        cost_mask_bce = _pairwise_bce_cost(pred_masks, target_masks)
        cost_mask_dice = _pairwise_dice_cost(pred_masks, target_masks)
        return (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
            + self.cost_mask_bce * cost_mask_bce
            + self.cost_mask_dice * cost_mask_dice
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: Sequence[dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        pred_masks = outputs["pred_masks"]

        if pred_logits.shape[0] != len(targets):
            raise ValueError("Batch size mismatch between outputs and targets.")

        assignments: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, target in enumerate(targets):
            batch_pred_logits = pred_logits[batch_index]
            batch_pred_boxes = pred_boxes[batch_index]
            batch_pred_masks = pred_masks[batch_index]
            target_labels = target["labels"].to(batch_pred_logits.device).long()
            target_spatial_size = resolve_target_spatial_size(
                target,
                fallback=tuple(int(value) for value in batch_pred_masks.shape[-2:]),
            )
            target_boxes = normalize_xyxy_boxes(
                target["boxes"].to(batch_pred_boxes.device),
                spatial_size=target_spatial_size,
            )
            target_masks = self._resize_target_masks(
                target["masks"].to(batch_pred_masks.device),
                spatial_size=batch_pred_masks.shape[-2:],
            )

            if batch_pred_logits.shape[0] == 0 or target_labels.numel() == 0:
                empty = torch.zeros((0,), dtype=torch.int64, device=batch_pred_logits.device)
                assignments.append((empty, empty))
                continue

            cost_matrix = self._compute_cost_matrix(
                pred_logits=batch_pred_logits,
                pred_boxes=batch_pred_boxes,
                pred_masks=batch_pred_masks,
                target_labels=target_labels,
                target_boxes=target_boxes,
                target_masks=target_masks,
            )
            pred_indices, target_indices = linear_sum_assignment(cost_matrix)
            assignments.append(
                (
                    pred_indices.to(batch_pred_logits.device),
                    target_indices.to(batch_pred_logits.device),
                )
            )

        return assignments
