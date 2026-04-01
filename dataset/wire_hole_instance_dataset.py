"""
COCO-style instance dataset for wire and interface-hole segmentation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

try:
    from pycocotools.coco import COCO
except ImportError:  # pragma: no cover
    COCO = None

__all__ = [
    "WireHoleInstanceDataset",
    "instance_collate_fn",
    "get_instance_dataloader",
]


def _resize_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resize_mask(mask_array: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(mask_array.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized.astype(np.uint8))


def _polygon_to_mask(height: int, width: int, segmentation: list[list[float]]) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        draw.polygon([(polygon[idx], polygon[idx + 1]) for idx in range(0, len(polygon), 2)], fill=1)
    return np.asarray(mask, dtype=np.uint8)


def _bbox_xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x_coord, y_coord, width, height = bbox
    return float(x_coord), float(y_coord), float(x_coord + width), float(y_coord + height)


def _clip_xyxy(box: np.ndarray, height: int, width: int) -> np.ndarray:
    box = box.copy()
    box[0::2] = np.clip(box[0::2], 0, width)
    box[1::2] = np.clip(box[1::2], 0, height)
    return box


def _box_area_xyxy(box: np.ndarray) -> float:
    return float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))


class WireHoleInstanceDataset(Dataset):
    class_names = ["background", "wire", "interface-hole"]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 1024,
        roi_prob: float = 0.4,
        roi_focus_prob: float = 0.7,
        roi_scale: float = 1.4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if COCO is None:
            raise ImportError("pycocotools is required for WireHoleInstanceDataset. Install with pip install pycocotools")

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.roi_prob = float(roi_prob) if split == "train" else 0.0
        self.roi_focus_prob = float(roi_focus_prob)
        self.roi_scale = float(roi_scale)
        self.seed = int(seed)

        annotation_path = self.data_root / "annotations" / f"instances_{split}.json"
        self.image_dir = self.data_root / "images" / split
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")

        self.coco = COCO(str(annotation_path))
        self.image_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_instances(self, image_id: int, height: int, width: int) -> list[dict[str, Any]]:
        instances = []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)):
            segmentation = ann.get("segmentation", [])
            if not segmentation:
                continue
            mask = _polygon_to_mask(height=height, width=width, segmentation=segmentation)
            bbox = np.array(_bbox_xywh_to_xyxy(ann["bbox"]), dtype=np.float32)
            instances.append(
                {
                    "bbox": bbox,
                    "label": int(ann["category_id"]),
                    "mask": mask,
                    "area": float(ann.get("area", mask.sum())),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "group": int(ann.get("group", -1)),
                }
            )
        return instances

    def _sample_crop_box(self, width: int, height: int, instances: list[dict[str, Any]], rng: random.Random) -> np.ndarray:
        if not instances or rng.random() >= self.roi_focus_prob:
            crop_size = min(width, height)
            crop_width = rng.randint(max(crop_size // 2, 32), crop_size)
            crop_height = rng.randint(max(crop_size // 2, 32), crop_size)
            x1 = rng.randint(0, max(width - crop_width, 0))
            y1 = rng.randint(0, max(height - crop_height, 0))
            return np.array([x1, y1, x1 + crop_width, y1 + crop_height], dtype=np.float32)

        candidate_weights = []
        for instance in instances:
            bbox = instance["bbox"]
            area = _box_area_xyxy(bbox)
            small_object_boost = 2.0 if instance["label"] == 2 or area < 4096.0 else 1.0
            elongated_boost = 1.5 if max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / max(min(bbox[2] - bbox[0], bbox[3] - bbox[1]), 1.0) > 3.0 else 1.0
            candidate_weights.append(small_object_boost * elongated_boost)

        focus_index = rng.choices(range(len(instances)), weights=candidate_weights, k=1)[0]
        focus_box = instances[focus_index]["bbox"]
        box_width = max(focus_box[2] - focus_box[0], 4.0)
        box_height = max(focus_box[3] - focus_box[1], 4.0)
        crop_width = min(width, max(int(round(box_width * self.roi_scale)), 96))
        crop_height = min(height, max(int(round(box_height * self.roi_scale)), 96))
        center_x = 0.5 * (focus_box[0] + focus_box[2])
        center_y = 0.5 * (focus_box[1] + focus_box[3])
        jitter_x = rng.uniform(-0.1, 0.1) * crop_width
        jitter_y = rng.uniform(-0.1, 0.1) * crop_height
        x1 = int(round(center_x - crop_width * 0.5 + jitter_x))
        y1 = int(round(center_y - crop_height * 0.5 + jitter_y))
        x1 = max(0, min(x1, width - crop_width))
        y1 = max(0, min(y1, height - crop_height))
        return np.array([x1, y1, x1 + crop_width, y1 + crop_height], dtype=np.float32)

    def _apply_crop(
        self,
        image: Image.Image,
        instances: list[dict[str, Any]],
        crop_box: np.ndarray,
    ) -> tuple[Image.Image, list[dict[str, Any]], tuple[int, int]]:
        x1, y1, x2, y2 = crop_box.astype(int).tolist()
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_instances = []
        crop_height = y2 - y1
        crop_width = x2 - x1

        for instance in instances:
            cropped_mask = instance["mask"][y1:y2, x1:x2]
            if cropped_mask.sum() == 0:
                continue
            ys, xs = np.where(cropped_mask > 0)
            bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
            cropped_instances.append(
                {
                    "bbox": bbox,
                    "label": instance["label"],
                    "mask": cropped_mask,
                    "area": float(cropped_mask.sum()),
                    "iscrowd": instance["iscrowd"],
                    "group": instance["group"],
                }
            )
        return cropped_image, cropped_instances, (crop_height, crop_width)

    def _pack_instances(
        self,
        instances: list[dict[str, Any]],
        sample_height: int,
        sample_width: int,
        scale_y: float,
        scale_x: float,
    ) -> dict[str, Any]:
        if not instances:
            empty_masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            empty_long = torch.zeros((0,), dtype=torch.long)
            empty_float = torch.zeros((0,), dtype=torch.float32)
            return {
                "boxes": empty_boxes,
                "labels": empty_long,
                "masks": empty_masks,
                "areas": empty_float,
                "iscrowd": empty_long,
                "groups": [],
            }

        mask_tensors = []
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        groups = []

        for instance in instances:
            scaled_box = instance["bbox"].astype(np.float32).copy()
            scaled_box[[0, 2]] *= scale_x
            scaled_box[[1, 3]] *= scale_y
            boxes.append(torch.from_numpy(scaled_box))
            labels.append(int(instance["label"]))
            mask_tensors.append(_resize_mask(instance["mask"], self.image_size))
            areas.append(float(instance["area"] * scale_x * scale_y))
            iscrowd.append(int(instance["iscrowd"]))
            groups.append(int(instance["group"]))

        return {
            "boxes": torch.stack(boxes, dim=0).float(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": torch.stack(mask_tensors, dim=0).to(torch.uint8),
            "areas": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.long),
            "groups": groups,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        full_height = int(image_info["height"])
        full_width = int(image_info["width"])
        instances = self._load_instances(image_id=image_id, height=full_height, width=full_width)

        rng = random.Random((self.seed + 1) * (index + 17))
        crop_box = None
        sample_size = (full_height, full_width)
        if self.roi_prob > 0 and rng.random() < self.roi_prob:
            crop_box = self._sample_crop_box(full_width, full_height, instances, rng)
            image, instances, sample_size = self._apply_crop(image, instances, crop_box)

        sample_height, sample_width = sample_size
        scale_y = self.image_size / max(sample_height, 1)
        scale_x = self.image_size / max(sample_width, 1)
        image_tensor = _resize_image(image, self.image_size)
        packed_instances = self._pack_instances(instances, sample_height, sample_width, scale_y, scale_x)

        sample = {
            "image": image_tensor,
            "original_size": sample_size,
            "full_image_size": (full_height, full_width),
            "processed_size": (self.image_size, self.image_size),
            "resize_scale": (scale_y, scale_x),
            "crop_box": None if crop_box is None else crop_box.tolist(),
            "image_id": int(image_id),
            "instances": packed_instances,
            "image_path": str(image_path),
        }
        return sample


def instance_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    return {
        "image": images,
        "original_size": [sample["original_size"] for sample in batch],
        "full_image_size": [sample["full_image_size"] for sample in batch],
        "processed_size": [sample["processed_size"] for sample in batch],
        "resize_scale": [sample["resize_scale"] for sample in batch],
        "crop_box": [sample["crop_box"] for sample in batch],
        "image_id": [sample["image_id"] for sample in batch],
        "instances": [sample["instances"] for sample in batch],
        "image_path": [sample["image_path"] for sample in batch],
    }


def get_instance_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    roi_prob: float = 0.4,
    roi_focus_prob: float = 0.7,
    roi_scale: float = 1.4,
    seed: int = 42,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    dataset = WireHoleInstanceDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        roi_prob=roi_prob,
        roi_focus_prob=roi_focus_prob,
        roi_scale=roi_scale,
        seed=seed,
    )
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (split == "train"),
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": (split == "train"),
        "collate_fn": instance_collate_fn,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(persistent_workers)
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**dataloader_kwargs)
