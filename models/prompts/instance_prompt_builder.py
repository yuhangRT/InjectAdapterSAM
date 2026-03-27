"""
Prompt builder for box + one positive point refinement.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

__all__ = ["InstancePromptBuilder"]


def _distance_peak(mask: torch.Tensor) -> tuple[float, float]:
    mask_array = mask.detach().cpu().numpy().astype(np.uint8)
    if not mask_array.any():
        return 0.0, 0.0
    distance = cv2.distanceTransform(mask_array, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(distance)
    return float(max_loc[0]), float(max_loc[1])


class InstancePromptBuilder:
    def __init__(self, expand_ratio: float = 1.10):
        self.expand_ratio = float(expand_ratio)

    def expand_box(self, box: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        x1, y1, x2, y2 = box.float()
        box_width = x2 - x1
        box_height = y2 - y1
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        new_width = box_width * self.expand_ratio
        new_height = box_height * self.expand_ratio
        height, width = image_size
        expanded = torch.tensor(
            [
                max(0.0, center_x - new_width * 0.5),
                max(0.0, center_y - new_height * 0.5),
                min(float(width), center_x + new_width * 0.5),
                min(float(height), center_y + new_height * 0.5),
            ],
            device=box.device,
            dtype=torch.float32,
        )
        return expanded

    def _hole_point(self, box: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = box.float()
        return torch.tensor([0.5 * (x1 + x2), 0.5 * (y1 + y2)], device=box.device, dtype=torch.float32)

    def _wire_point(
        self,
        *,
        box: torch.Tensor,
        gt_mask: torch.Tensor | None,
        proposal_center: torch.Tensor | None,
    ) -> torch.Tensor:
        if gt_mask is not None and int(gt_mask.sum().item()) > 0:
            peak_x, peak_y = _distance_peak(gt_mask)
            return torch.tensor([peak_x, peak_y], device=box.device, dtype=torch.float32)
        if proposal_center is not None:
            return proposal_center.to(device=box.device, dtype=torch.float32)
        return self._hole_point(box)

    def build_prompt(
        self,
        *,
        category_id: int,
        box: torch.Tensor,
        image_size: tuple[int, int],
        gt_mask: torch.Tensor | None = None,
        proposal_center: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        expanded_box = self.expand_box(box, image_size=image_size)
        if int(category_id) == 2:
            point = self._hole_point(box)
        else:
            point = self._wire_point(box=box, gt_mask=gt_mask, proposal_center=proposal_center)
        return {
            "boxes": expanded_box.unsqueeze(0),
            "point_coords": point.view(1, 1, 2),
            "point_labels": torch.ones((1, 1), device=box.device, dtype=torch.int64),
            "source_prompt": "box+pos1",
        }

    def build_batch_prompts(
        self,
        proposals: list[dict[str, Any]],
        *,
        image_size: tuple[int, int],
    ) -> dict[str, Any]:
        if not proposals:
            empty = torch.zeros((0, 4), dtype=torch.float32)
            return {
                "boxes": empty,
                "point_coords": torch.zeros((0, 1, 2), dtype=torch.float32),
                "point_labels": torch.zeros((0, 1), dtype=torch.int64),
                "source_prompt": [],
            }

        boxes = []
        point_coords = []
        point_labels = []
        source_prompt = []
        device = proposals[0]["bbox"].device
        for proposal in proposals:
            prompt = self.build_prompt(
                category_id=int(proposal["category_id"]),
                box=proposal["bbox"],
                image_size=image_size,
                gt_mask=proposal.get("gt_mask"),
                proposal_center=proposal.get("center"),
            )
            boxes.append(prompt["boxes"][0])
            point_coords.append(prompt["point_coords"][0])
            point_labels.append(prompt["point_labels"][0])
            source_prompt.append(prompt["source_prompt"])
        return {
            "boxes": torch.stack(boxes, dim=0).to(device=device),
            "point_coords": torch.stack(point_coords, dim=0).to(device=device),
            "point_labels": torch.stack(point_labels, dim=0).to(device=device),
            "source_prompt": source_prompt,
        }
