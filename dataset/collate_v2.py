"""Batch collation for the S02 label/hole instance dataset."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["collate_label_hole_batch"]


def collate_label_hole_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("collate_label_hole_batch expects a non-empty batch")
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    return {
        "image": images,
        "image_id": [int(sample["image_id"]) for sample in batch],
        "image_path": [sample["image_path"] for sample in batch],
        "orig_size": [tuple(sample["orig_size"]) for sample in batch],
        "processed_size": [tuple(sample["processed_size"]) for sample in batch],
        "crop_box": [sample["crop_box"] for sample in batch],
        "crop_mode": [sample.get("crop_mode") for sample in batch],
        "instances": [sample["instances"] for sample in batch],
    }
