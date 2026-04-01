"""
CenterNet-lite anchor-free proposal head.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.crnet_blocks import ConvBN

__all__ = ["CenterNetLiteProposalHead"]


def _gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> int:
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2
    return max(0, int(r1))


def _draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    if radius <= 0:
        x_coord, y_coord = center
        if 0 <= y_coord < heatmap.shape[0] and 0 <= x_coord < heatmap.shape[1]:
            heatmap[y_coord, x_coord] = max(heatmap[y_coord, x_coord], 1.0)
        return
    diameter = 2 * radius + 1
    gaussian = cv2.getGaussianKernel(diameter, diameter / 6)
    gaussian = np.outer(gaussian, gaussian)
    x_coord, y_coord = center
    left, right = min(x_coord, radius), min(heatmap.shape[1] - x_coord - 1, radius)
    top, bottom = min(y_coord, radius), min(heatmap.shape[0] - y_coord - 1, radius)
    masked_heatmap = heatmap[y_coord - top : y_coord + bottom + 1, x_coord - left : x_coord + right + 1]
    masked_gaussian = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    if masked_heatmap.size and masked_gaussian.size:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def _mask_distance_peak(mask: np.ndarray) -> tuple[float, float]:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(distance)
    return float(max_loc[0]), float(max_loc[1])


def _compute_center(mask: torch.Tensor, box: torch.Tensor, label: int) -> tuple[float, float]:
    mask_array = mask.detach().cpu().numpy().astype(np.uint8)
    if label == 1 and mask_array.any():
        return _mask_distance_peak(mask_array)
    x1, y1, x2, y2 = box.detach().cpu().tolist()
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> list[int]:
    if boxes.numel() == 0:
        return []
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = int(order[0])
        keep.append(current)
        if order.numel() == 1:
            break
        ious = _box_iou(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


class CenterNetLiteProposalHead(nn.Module):
    def __init__(self, in_channels: int = 256, feat_channels: int = 128, num_classes: int = 2, stride: int = 8):
        super().__init__()
        self.num_classes = int(num_classes)
        self.stride = int(stride)
        self.fuse_c3 = ConvBN(in_channels, feat_channels, 3)
        self.fuse_c4 = ConvBN(in_channels, feat_channels, 3)
        self.fuse_c5 = ConvBN(in_channels, feat_channels, 3)
        self.merge_conv = ConvBN(feat_channels * 3, feat_channels, 1)
        self.fuse = nn.Sequential(
            ConvBN(feat_channels, feat_channels, 3),
            nn.GELU(),
            nn.ConvTranspose2d(feat_channels, feat_channels, kernel_size=2, stride=2),
            nn.GELU(),
            ConvBN(feat_channels, feat_channels, 3),
            nn.GELU(),
        )
        self.center_head = nn.Conv2d(feat_channels, num_classes, kernel_size=1)
        self.size_head = nn.Conv2d(feat_channels, 2, kernel_size=1)
        self.offset_head = nn.Conv2d(feat_channels, 2, kernel_size=1)
        nn.init.constant_(self.center_head.bias, -2.19)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fused = torch.cat([self.fuse_c3(features["c3"]), self.fuse_c4(features["c4"]), self.fuse_c5(features["c5"])], dim=1)
        fused = self.merge_conv(fused)
        fused = self.fuse(fused)
        return {
            "center_heatmap": self.center_head(fused),
            "size_map": F.relu(self.size_head(fused)),
            "offset_map": self.offset_head(fused),
            "proposal_features": fused,
        }

    def build_targets(
        self,
        instances: list[dict[str, Any]],
        *,
        image_size: int,
        hole_positive_weight: float = 1.5,
    ) -> dict[str, torch.Tensor]:
        target_size = image_size // self.stride
        batch_size = len(instances)
        heatmap = torch.zeros((batch_size, self.num_classes, target_size, target_size), dtype=torch.float32)
        size_target = torch.zeros((batch_size, 2, target_size, target_size), dtype=torch.float32)
        offset_target = torch.zeros((batch_size, 2, target_size, target_size), dtype=torch.float32)
        positive_mask = torch.zeros((batch_size, target_size, target_size), dtype=torch.bool)
        offset_mask = torch.zeros((batch_size, target_size, target_size), dtype=torch.bool)
        positive_weight = torch.ones((batch_size, target_size, target_size), dtype=torch.float32)

        for batch_idx, sample_instances in enumerate(instances):
            for box, label, mask in zip(sample_instances["boxes"], sample_instances["labels"], sample_instances["masks"]):
                center_x, center_y = _compute_center(mask, box, int(label.item()))
                center_x = center_x / self.stride
                center_y = center_y / self.stride
                grid_x = int(center_x)
                grid_y = int(center_y)
                if not (0 <= grid_x < target_size and 0 <= grid_y < target_size):
                    continue
                box_width = float((box[2] - box[0]).item() / self.stride)
                box_height = float((box[3] - box[1]).item() / self.stride)
                radius = _gaussian_radius(box_height, box_width)
                if int(label.item()) == 2:
                    radius = max(radius, 2)
                heatmap_np = heatmap[batch_idx, int(label.item()) - 1].numpy()
                _draw_gaussian(heatmap_np, (grid_x, grid_y), radius)
                heatmap[batch_idx, int(label.item()) - 1] = torch.from_numpy(heatmap_np)
                size_target[batch_idx, :, grid_y, grid_x] = torch.tensor([box_width, box_height], dtype=torch.float32)
                offset_target[batch_idx, :, grid_y, grid_x] = torch.tensor(
                    [center_x - grid_x, center_y - grid_y],
                    dtype=torch.float32,
                )
                positive_mask[batch_idx, grid_y, grid_x] = True
                offset_mask[batch_idx, grid_y, grid_x] = True
                if int(label.item()) == 2:
                    positive_weight[batch_idx, grid_y, grid_x] = float(hole_positive_weight)

        return {
            "heatmap": heatmap,
            "size_target": size_target,
            "offset_target": offset_target,
            "positive_mask": positive_mask,
            "offset_mask": offset_mask,
            "positive_weight": positive_weight,
        }

    @torch.no_grad()
    def decode(
        self,
        predictions: dict[str, torch.Tensor],
        *,
        image_size: int,
        topk_per_class: int = 64,
        box_nms_iou: float = 0.5,
    ) -> list[list[dict[str, torch.Tensor | float | int]]]:
        center_heatmap = torch.sigmoid(predictions["center_heatmap"])
        center_heatmap = center_heatmap * (center_heatmap == F.max_pool2d(center_heatmap, kernel_size=3, stride=1, padding=1))
        size_map = predictions["size_map"]
        offset_map = predictions["offset_map"]
        batch_results = []

        for batch_idx in range(center_heatmap.shape[0]):
            image_results = []
            for class_idx in range(self.num_classes):
                scores = center_heatmap[batch_idx, class_idx].reshape(-1)
                topk = min(topk_per_class, scores.numel())
                top_scores, top_indices = torch.topk(scores, k=topk)
                ys = top_indices // center_heatmap.shape[-1]
                xs = top_indices % center_heatmap.shape[-1]
                boxes = []
                proposals = []
                for score, x_idx, y_idx in zip(top_scores, xs, ys):
                    width_height = size_map[batch_idx, :, y_idx, x_idx]
                    offsets = offset_map[batch_idx, :, y_idx, x_idx]
                    center_x = (x_idx.float() + offsets[0]) * self.stride
                    center_y = (y_idx.float() + offsets[1]) * self.stride
                    box_width = width_height[0] * self.stride
                    box_height = width_height[1] * self.stride
                    x1 = torch.clamp(center_x - box_width * 0.5, min=0.0, max=float(image_size))
                    y1 = torch.clamp(center_y - box_height * 0.5, min=0.0, max=float(image_size))
                    x2 = torch.clamp(center_x + box_width * 0.5, min=0.0, max=float(image_size))
                    y2 = torch.clamp(center_y + box_height * 0.5, min=0.0, max=float(image_size))
                    box = torch.tensor([x1, y1, x2, y2], device=score.device, dtype=torch.float32)
                    boxes.append(box)
                    proposals.append(
                        {
                            "category_id": class_idx + 1,
                            "score": float(score.item()),
                            "bbox": box,
                            "center": torch.tensor([center_x, center_y], device=score.device, dtype=torch.float32),
                        }
                    )

                if boxes:
                    boxes_tensor = torch.stack(boxes, dim=0)
                    scores_tensor = torch.tensor([proposal["score"] for proposal in proposals], device=boxes_tensor.device)
                    keep_indices = _nms_boxes(boxes_tensor, scores_tensor, box_nms_iou)
                    image_results.extend([proposals[idx] for idx in keep_indices])
            batch_results.append(sorted(image_results, key=lambda item: item["score"], reverse=True))
        return batch_results
