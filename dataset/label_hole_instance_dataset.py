"""Label/hole instance dataset for the S02 pipeline."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .geometry_utils import (
    bbox_xyxy_from_mask,
    multi_polygon_to_mask,
    oriented_box_from_mask,
    principal_axis_from_mask,
    normalize_polygon,
)
from .collate_v2 import collate_label_hole_batch
from .transforms_v2 import StableInstanceTransforms, build_stable_transforms

try:
    from pycocotools.coco import COCO
except ImportError as exc:  # pragma: no cover - environment should provide official pycocotools
    raise ImportError("pycocotools is required for LabelHoleInstanceDataset") from exc

__all__ = [
    "LabelHoleInstanceDataset",
    "build_label_hole_dataloader",
]


class LabelHoleInstanceDataset(Dataset):
    """COCO-based instance dataset for `label_sleeve` and `empty_terminal`."""

    class_names = ["background", "label_sleeve", "empty_terminal"]

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int = 1024,
        full_image_prob: float = 0.5,
        object_crop_prob: float = 0.5,
        hole_focused_prob: float = 0.6,
        label_focused_prob: float = 0.4,
        hole_scale_range: tuple[float, float] = (2.2, 3.2),
        label_scale_range: tuple[float, float] = (1.4, 2.2),
        object_scale_range: tuple[float, float] = (1.8, 2.6),
        min_crop_size: int = 96,
        seed: int = 42,
        augment: bool | None = None,
        transform: StableInstanceTransforms | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.full_image_prob = float(full_image_prob)
        self.object_crop_prob = float(object_crop_prob)
        self.hole_focused_prob = float(hole_focused_prob)
        self.label_focused_prob = float(label_focused_prob)
        self.hole_scale_range = tuple(float(value) for value in hole_scale_range)
        self.label_scale_range = tuple(float(value) for value in label_scale_range)
        self.object_scale_range = tuple(float(value) for value in object_scale_range)
        self.min_crop_size = int(min_crop_size)
        self.seed = int(seed)
        self.augment = bool(split == "train") if augment is None else bool(augment)

        annotation_path = self.data_root / "annotations" / f"instances_{split}.json"
        image_dir = self.data_root / "images" / split
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.coco = COCO(str(annotation_path))
        self.image_ids = sorted(int(image_id) for image_id in self.coco.getImgIds())

        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_stable_transforms(
                image_size=self.image_size,
                rotate_prob=0.18 if self.augment else 0.0,
                rotate_deg=2.0,
                brightness_range=(0.92, 1.08) if self.augment else (1.0, 1.0),
                contrast_range=(0.92, 1.08) if self.augment else (1.0, 1.0),
                blur_prob=0.12 if self.augment else 0.0,
                blur_radius=(0.6, 1.0),
                jpeg_prob=0.15 if self.augment else 0.0,
                jpeg_quality=(78, 95),
                noise_prob=0.12 if self.augment else 0.0,
                noise_std=(0.0, 0.01),
                pad_fill=(0, 0, 0),
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_image_info(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        return self.coco.loadImgs([image_id])[0]

    def load_original_instances(self, image_id: int) -> list[dict[str, Any]]:
        image_info = self.coco.loadImgs([int(image_id)])[0]
        height = int(image_info["height"])
        width = int(image_info["width"])
        ann_ids = self.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)
        instances: list[dict[str, Any]] = []
        for ann in annotations:
            category_id = int(ann.get("category_id", 0))
            if category_id not in (1, 2):
                raise ValueError(f"Unsupported category id {category_id} in {self.annotation_path}")
            segmentation = ann.get("segmentation", [])
            polygons = self._segmentation_to_polygons(segmentation)
            if not polygons:
                continue
            mask = np.asarray(multi_polygon_to_mask(height, width, polygons), dtype=np.uint8)
            if mask.sum() == 0:
                continue
            instances.append(
                {
                    "label": category_id,
                    "mask": mask,
                    "area": float(mask.sum()),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "group_id": int(ann.get("group_id", ann.get("group", -1))),
                    "ann_id": int(ann.get("id", -1)),
                    "bbox_xyxy": np.asarray(bbox_xyxy_from_mask(mask.tolist()), dtype=np.float32),
                    "oriented_box": np.asarray(oriented_box_from_mask(mask.tolist()), dtype=np.float32),
                    "principal_axis": np.asarray(principal_axis_from_mask(mask.tolist()), dtype=np.float32),
                }
            )
        return instances

    @staticmethod
    def _segmentation_to_polygons(segmentation: object) -> list[list[tuple[float, float]]]:
        if not segmentation:
            return []
        if isinstance(segmentation, dict):
            return []
        polygons: list[list[tuple[float, float]]] = []
        if isinstance(segmentation, list):
            if segmentation and isinstance(segmentation[0], (int, float)):
                points = normalize_polygon(segmentation)
                if len(points) >= 3:
                    polygons.append(points)
                return polygons
            for polygon in segmentation:
                points = normalize_polygon(polygon)
                if len(points) >= 3:
                    polygons.append(points)
            return polygons
        return []

    @staticmethod
    def _union_bbox(instances: list[dict[str, Any]]) -> np.ndarray:
        xs1 = [float(instance["bbox_xyxy"][0]) for instance in instances]
        ys1 = [float(instance["bbox_xyxy"][1]) for instance in instances]
        xs2 = [float(instance["bbox_xyxy"][2]) for instance in instances]
        ys2 = [float(instance["bbox_xyxy"][3]) for instance in instances]
        return np.asarray([min(xs1), min(ys1), max(xs2), max(ys2)], dtype=np.float32)

    def _select_instance(
        self,
        instances: list[dict[str, Any]],
        labels: set[int] | None,
        rng: random.Random,
    ) -> dict[str, Any] | None:
        candidates = [instance for instance in instances if labels is None or int(instance["label"]) in labels]
        if not candidates:
            return None
        weights = []
        for instance in candidates:
            area = max(float(instance["area"]), 1.0)
            bbox = instance["bbox_xyxy"]
            width = max(float(bbox[2] - bbox[0]), 1.0)
            height = max(float(bbox[3] - bbox[1]), 1.0)
            aspect = max(width / height, height / width)
            weight = 1.0 / math.sqrt(area)
            if int(instance["label"]) == 2:
                weight *= 2.5
            if aspect > 3.0:
                weight *= 1.5
            weights.append(weight)
        return rng.choices(candidates, weights=weights, k=1)[0]

    def _expand_crop(
        self,
        bbox_xyxy: np.ndarray,
        scale_range: tuple[float, float],
        image_width: int,
        image_height: int,
        rng: random.Random,
    ) -> np.ndarray:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy.tolist()]
        bbox_width = max(x2 - x1, 1.0)
        bbox_height = max(y2 - y1, 1.0)
        scale = float(rng.uniform(*scale_range))
        crop_width = max(int(round(bbox_width * scale)), self.min_crop_size)
        crop_height = max(int(round(bbox_height * scale)), self.min_crop_size)
        crop_width = min(crop_width, image_width)
        crop_height = min(crop_height, image_height)
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        jitter_x = rng.uniform(-0.08, 0.08) * crop_width
        jitter_y = rng.uniform(-0.08, 0.08) * crop_height
        crop_x1 = int(round(center_x - crop_width * 0.5 + jitter_x))
        crop_y1 = int(round(center_y - crop_height * 0.5 + jitter_y))
        crop_x1 = max(0, min(crop_x1, image_width - crop_width))
        crop_y1 = max(0, min(crop_y1, image_height - crop_height))
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height
        return np.asarray([crop_x1, crop_y1, crop_x2, crop_y2], dtype=np.float32)

    def _sample_crop_box(
        self,
        image_width: int,
        image_height: int,
        instances: list[dict[str, Any]],
        rng: random.Random,
    ) -> tuple[np.ndarray | None, str]:
        if not instances or rng.random() < self.full_image_prob:
            return None, "full"

        if rng.random() < self.object_crop_prob:
            selected = self._select_instance(instances, labels=None, rng=rng)
            if selected is None:
                return None, "full"
            target_instances = [selected]
            scale_range = self.object_scale_range
            mode = "object"
        else:
            focus_total = max(self.hole_focused_prob + self.label_focused_prob, 0.0)
            if focus_total <= 0.0:
                selected = self._select_instance(instances, labels=None, rng=rng)
                if selected is None:
                    return None, "full"
                target_instances = [selected]
                scale_range = self.object_scale_range
                mode = "object"
            else:
                focus_roll = rng.random() * focus_total
                if focus_roll < self.hole_focused_prob:
                    selected = self._select_instance(instances, labels={2}, rng=rng)
                    if selected is None:
                        selected = self._select_instance(instances, labels=None, rng=rng)
                    mode = "hole"
                    scale_range = self.hole_scale_range
                else:
                    selected = self._select_instance(instances, labels={1}, rng=rng)
                    if selected is None:
                        selected = self._select_instance(instances, labels=None, rng=rng)
                    mode = "label"
                    scale_range = self.label_scale_range
                if selected is None:
                    return None, "full"
                selected_group = int(selected.get("group_id", -1))
                if selected_group >= 0:
                    target_instances = [
                        instance
                        for instance in instances
                        if int(instance.get("group_id", -1)) == selected_group
                        and int(instance["label"]) == int(selected["label"])
                    ]
                    if not target_instances:
                        target_instances = [selected]
                else:
                    target_instances = [selected]

        bbox = self._union_bbox(target_instances)
        crop_box = self._expand_crop(bbox, scale_range, image_width, image_height, rng)
        return crop_box, mode

    @staticmethod
    def _crop_image(image: Image.Image, crop_box: np.ndarray) -> Image.Image:
        x1, y1, x2, y2 = [int(round(value)) for value in crop_box.tolist()]
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _crop_instances(instances: list[dict[str, Any]], crop_box: np.ndarray) -> list[dict[str, Any]]:
        x1, y1, x2, y2 = [int(round(value)) for value in crop_box.tolist()]
        cropped_instances: list[dict[str, Any]] = []
        for instance in instances:
            cropped_mask = np.asarray(instance["mask"], dtype=np.uint8)[y1:y2, x1:x2]
            if cropped_mask.sum() == 0:
                continue
            cropped_instances.append(
                {
                    "label": int(instance["label"]),
                    "mask": cropped_mask,
                    "area": float(cropped_mask.sum()),
                    "iscrowd": int(instance["iscrowd"]),
                    "group_id": int(instance["group_id"]),
                }
            )
        return cropped_instances

    @staticmethod
    def _finalize_instances(instances: list[dict[str, Any]], mask_tensor: torch.Tensor) -> dict[str, Any]:
        num_instances = int(mask_tensor.shape[0])
        if num_instances == 0:
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            empty_labels = torch.zeros((0,), dtype=torch.long)
            empty_masks = torch.zeros((0, mask_tensor.shape[-2], mask_tensor.shape[-1]), dtype=torch.uint8)
            empty_float = torch.zeros((0,), dtype=torch.float32)
            empty_long = torch.zeros((0,), dtype=torch.long)
            return {
                "boxes": empty_boxes,
                "labels": empty_labels,
                "masks": empty_masks,
                "areas": empty_float,
                "iscrowd": empty_long,
                "group_ids": empty_long,
                "oriented_boxes": torch.zeros((0, 5), dtype=torch.float32),
                "principal_axes": torch.zeros((0, 2), dtype=torch.float32),
            }

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        group_ids = []
        oriented_boxes = []
        principal_axes = []
        for index, instance in enumerate(instances):
            mask = mask_tensor[index].cpu().numpy().tolist()
            boxes.append(torch.tensor(bbox_xyxy_from_mask(mask), dtype=torch.float32))
            labels.append(int(instance["label"]))
            areas.append(float(mask_tensor[index].sum().item()))
            iscrowd.append(int(instance["iscrowd"]))
            group_ids.append(int(instance["group_id"]))
            oriented_boxes.append(torch.tensor(oriented_box_from_mask(mask), dtype=torch.float32))
            principal_axes.append(torch.tensor(principal_axis_from_mask(mask), dtype=torch.float32))
        return {
            "boxes": torch.stack(boxes, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": mask_tensor.to(torch.uint8),
            "areas": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.long),
            "group_ids": torch.tensor(group_ids, dtype=torch.long),
            "oriented_boxes": torch.stack(oriented_boxes, dim=0),
            "principal_axes": torch.stack(principal_axes, dim=0),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.image_dir / image_info["file_name"]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        orig_size = (int(image_info["height"]), int(image_info["width"]))
        instances = self.load_original_instances(image_id)

        rng = random.Random((self.seed + 17) * (index + 1))
        crop_box, crop_mode = self._sample_crop_box(orig_size[1], orig_size[0], instances, rng)
        if crop_box is not None:
            image = self._crop_image(image, crop_box)
            instances = self._crop_instances(instances, crop_box)

        image_tensor, mask_tensor, _ = self.transform(image, [instance["mask"] for instance in instances], np.random.default_rng(self.seed + index))
        finalized = self._finalize_instances(instances, mask_tensor)

        return {
            "image": image_tensor,
            "image_id": int(image_id),
            "image_path": str(image_path),
            "orig_size": orig_size,
            "processed_size": (self.image_size, self.image_size),
            "crop_box": None if crop_box is None else [float(value) for value in crop_box.tolist()],
            "crop_mode": crop_mode,
            "instances": finalized,
        }


def build_label_hole_dataloader(
    data_root: str | Path,
    split: str = "train",
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = LabelHoleInstanceDataset(data_root=data_root, split=split, **dataset_kwargs)
    effective_persistent_workers = bool(persistent_workers and num_workers > 0)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (split == "train"),
        "num_workers": num_workers,
        "collate_fn": collate_label_hole_batch,
        "pin_memory": bool(pin_memory),
        "drop_last": (split == "train"),
        "persistent_workers": effective_persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(
        **loader_kwargs,
    )
