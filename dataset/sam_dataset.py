"""
Dataset loaders for WireCR-SAM experiments.

`wire_hole` is the primary industrial 3-class semantic segmentation task.
`coco` is an auxiliary category-agnostic foreground segmentation task used for
smoke tests or transfer experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from pycocotools.coco import COCO
except ImportError:  # pragma: no cover - optional unless --dataset coco is used
    COCO = None

__all__ = [
    "WireHoleDataset",
    "CocoForegroundDataset",
    "get_dataset_class_names",
    "validate_dataset_config",
    "get_sam_dataloader",
]


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COCO_IMAGE_DIRS = {
    "train": ("train2017", "images/train2017"),
    "val": ("val2017", "images/val2017"),
    "test": ("test2017", "images/test2017"),
}
COCO_ANNOTATIONS = {
    "train": ("annotations/instances_train2017.json",),
    "val": ("annotations/instances_val2017.json",),
    "test": ("annotations/instances_test2017.json",),
}


def get_dataset_class_names(dataset_name: str, num_classes: int) -> List[str]:
    if dataset_name == "wire_hole":
        base_names = ["background", "wire", "hole"]
    elif dataset_name == "coco":
        base_names = ["background", "foreground"]
    else:
        base_names = ["background"] + [f"class_{idx}" for idx in range(1, num_classes)]

    if len(base_names) >= num_classes:
        return base_names[:num_classes]
    return base_names + [f"class_{idx}" for idx in range(len(base_names), num_classes)]


def validate_dataset_config(dataset_name: str, num_classes: int) -> None:
    expected_classes = {"wire_hole": 3, "coco": 2}
    if dataset_name not in expected_classes:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets are: {sorted(expected_classes)}."
        )
    if num_classes != expected_classes[dataset_name]:
        raise ValueError(
            f"Dataset '{dataset_name}' expects --num-classes {expected_classes[dataset_name]}, "
            f"but got {num_classes}."
        )


def _sample_subset(samples, subset_ratio: float, subset_seed: int):
    if subset_ratio >= 1.0:
        return samples
    if subset_ratio <= 0:
        raise ValueError("subset_ratio must be in (0, 1]")

    generator = np.random.default_rng(subset_seed)
    indices = np.arange(len(samples))
    generator.shuffle(indices)
    keep = max(1, int(round(len(indices) * subset_ratio)))
    return [samples[idx] for idx in indices[:keep]]


def _resolve_split_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    candidates = [
        (root / "images" / split, root / "masks" / split),
        (root / "images" / split, root / "labels" / split),
        (root / split / "images", root / split / "masks"),
        (root / split / "images", root / split / "labels"),
    ]

    for image_dir, mask_dir in candidates:
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir, mask_dir

    raise FileNotFoundError(
        f"Could not find image/mask directories for split '{split}' under {root}. "
        "Expected one of: images/<split> + masks/<split>, images/<split> + labels/<split>, "
        "<split>/images + <split>/masks, or <split>/images + <split>/labels."
    )


def _resolve_coco_split(root: Path, split: str) -> Tuple[Path, Path]:
    annotation_candidates = [root / rel_path for rel_path in COCO_ANNOTATIONS.get(split, ())]
    image_candidates = [root / rel_path for rel_path in COCO_IMAGE_DIRS.get(split, ())]

    for annotation_file in annotation_candidates:
        for image_dir in image_candidates:
            if annotation_file.is_file() and image_dir.is_dir():
                return annotation_file, image_dir

    raise FileNotFoundError(
        f"Could not find COCO files for split '{split}' under {root}. "
        "Expected annotations/instances_<split>2017.json and either <split>2017/ or images/<split>2017/."
    )


def _build_mask_index(mask_dir: Path) -> Dict[str, Path]:
    index = {}
    for path in sorted(mask_dir.iterdir()):
        if path.is_file():
            index[path.stem] = path
    return index


def _list_samples(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    mask_index = _build_mask_index(mask_dir)
    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = mask_index.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for image: {image_path.name}")
        samples.append((image_path, mask_path))

    if not samples:
        raise FileNotFoundError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    return samples


def _resize_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()


def _remap_mask_values(mask_array: np.ndarray, num_classes: int) -> np.ndarray:
    if mask_array.ndim == 3:
        flat = mask_array.reshape(-1, mask_array.shape[-1])
        unique_values = np.unique(flat, axis=0)
        if unique_values.shape[0] > num_classes:
            raise ValueError(
                f"Found {unique_values.shape[0]} unique mask colors, expected at most {num_classes}"
            )
        mapping = {
            tuple(value.tolist()): idx
            for idx, value in enumerate(sorted(unique_values.tolist()))
        }
        remapped = np.zeros(mask_array.shape[:2], dtype=np.int64)
        for color, idx in mapping.items():
            matches = np.all(mask_array == np.asarray(color, dtype=mask_array.dtype), axis=-1)
            remapped[matches] = idx
        return remapped

    unique_values = sorted(np.unique(mask_array).tolist())
    if len(unique_values) > num_classes:
        raise ValueError(f"Found {len(unique_values)} mask labels, expected at most {num_classes}")
    if unique_values == list(range(len(unique_values))):
        return mask_array.astype(np.int64)

    mapping = {value: idx for idx, value in enumerate(unique_values)}
    remapped = np.zeros_like(mask_array, dtype=np.int64)
    for value, idx in mapping.items():
        remapped[mask_array == value] = idx
    return remapped


def _resize_mask(mask: Image.Image, image_size: int, num_classes: int) -> torch.Tensor:
    resized = mask.resize((image_size, image_size), resample=Image.NEAREST)
    mask_array = np.asarray(resized)
    remapped = _remap_mask_values(mask_array, num_classes=num_classes)
    return torch.from_numpy(remapped).long()


class WireHoleDataset(Dataset):
    """
    Dataset for industrial wire / interface-hole semantic segmentation.

    Expected labels:
      0 -> background
      1 -> wire
      2 -> interface-hole
    """

    class_names = ["background", "wire", "hole"]

    def __init__(
        self,
        data_root,
        split="train",
        subset_ratio=1.0,
        subset_seed=42,
        image_size=1024,
        num_classes=3,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes

        image_dir, mask_dir = _resolve_split_dirs(self.data_root, split)
        samples = _list_samples(image_dir, mask_dir)

        if split == "train":
            samples = _sample_subset(samples, subset_ratio, subset_seed)

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        original_size = image.size[::-1]
        image_tensor = _resize_image(image, self.image_size)
        mask_tensor = _resize_mask(mask, self.image_size, self.num_classes)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "original_size": original_size,
            "output_size": (self.image_size, self.image_size),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


class CocoForegroundDataset(Dataset):
    """Auxiliary category-agnostic COCO dataset for foreground/background segmentation."""

    class_names = ["background", "foreground"]

    def __init__(
        self,
        data_root,
        split="train",
        subset_ratio=1.0,
        subset_seed=42,
        image_size=1024,
        num_classes=2,
    ):
        super().__init__()

        if COCO is None:
            raise ImportError(
                "pycocotools is required for --dataset coco. Install it with: pip install pycocotools"
            )

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes

        annotation_file, image_dir = _resolve_coco_split(self.data_root, split)
        self.image_dir = image_dir
        self.coco = COCO(str(annotation_file))

        image_ids = sorted(self.coco.getImgIds())
        if not image_ids:
            raise FileNotFoundError(f"No COCO images found in annotation file: {annotation_file}")

        if split == "train":
            image_ids = _sample_subset(image_ids, subset_ratio, subset_seed)

        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def _build_foreground_mask(self, image_id: int, height: int, width: int) -> np.ndarray:
        merged_mask = np.zeros((height, width), dtype=np.uint8)
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
        for annotation in self.coco.loadAnns(annotation_ids):
            merged_mask = np.maximum(merged_mask, self.coco.annToMask(annotation).astype(np.uint8))
        return merged_mask

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs([image_id])[0]

        image_path = self.image_dir / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {image_path}")

        image = Image.open(image_path)
        original_size = image.size[::-1]

        foreground_mask = self._build_foreground_mask(
            image_id=image_id,
            height=image_info["height"],
            width=image_info["width"],
        )
        mask = Image.fromarray(foreground_mask, mode="L")

        image_tensor = _resize_image(image, self.image_size)
        mask_tensor = _resize_mask(mask, self.image_size, self.num_classes)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "original_size": original_size,
            "output_size": (self.image_size, self.image_size),
            "image_path": str(image_path),
            "image_id": image_id,
        }


def get_sam_dataloader(
    data_root,
    split="train",
    dataset_name="wire_hole",
    batch_size=4,
    num_workers=4,
    image_size=1024,
    subset_ratio=1.0,
    subset_seed=42,
    num_classes=3,
):
    validate_dataset_config(dataset_name, num_classes)

    dataset_cls = {
        "wire_hole": WireHoleDataset,
        "coco": CocoForegroundDataset,
    }[dataset_name]

    dataset = dataset_cls(
        data_root=data_root,
        split=split,
        subset_ratio=subset_ratio if split == "train" else 1.0,
        subset_seed=subset_seed,
        image_size=image_size,
        num_classes=num_classes,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
