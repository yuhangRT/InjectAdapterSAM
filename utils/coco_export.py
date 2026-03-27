"""
Utilities for exporting ISAT polygons to COCO-style instance annotations.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CATEGORY_NAME_TO_ID = {
    "wire": 1,
    "hole": 2,
    "interface-hole": 2,
}
CATEGORIES = [
    {"id": 1, "name": "wire", "supercategory": "foreground"},
    {"id": 2, "name": "interface-hole", "supercategory": "foreground"},
]


@dataclass(frozen=True)
class GroupedInstance:
    image_stem: str
    image_path: Path
    annotation_path: Path
    width: int
    height: int
    category_id: int
    category_name: str
    group: int
    polygons: tuple[list[float], ...]
    area: float
    bbox: tuple[float, float, float, float]
    polygon_count: int


class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def resolve_image_path(src_root: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = src_root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find an image file for stem '{stem}' under {src_root}.")


def discover_isat_samples(src_root: Path) -> list[dict[str, Path]]:
    manifest_path = src_root / "labeled_manifest.csv"
    if manifest_path.is_file():
        samples = []
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = src_root / row["image"]
                annotation_path = src_root / row["annotation"]
                samples.append(
                    {
                        "stem": row["base"],
                        "image_path": image_path,
                        "annotation_path": annotation_path,
                    }
                )
        return samples

    samples = []
    for annotation_path in sorted(src_root.glob("*.json")):
        stem = annotation_path.stem
        samples.append(
            {
                "stem": stem,
                "image_path": resolve_image_path(src_root, stem),
                "annotation_path": annotation_path,
            }
        )
    if not samples:
        raise FileNotFoundError(f"No ISAT JSON annotations found under {src_root}")
    return samples


def load_isat_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def canonical_category_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if name not in CATEGORY_NAME_TO_ID:
        raise ValueError(f"Unsupported category '{raw_name}'. Expected one of {sorted(CATEGORY_NAME_TO_ID)}.")
    return "interface-hole" if CATEGORY_NAME_TO_ID[name] == 2 else "wire"


def normalize_polygon(segmentation: object) -> list[tuple[float, float]]:
    if not segmentation:
        return []
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        if len(segmentation[0]) == 2:
            return [(float(x), float(y)) for x, y in segmentation]
        if len(segmentation) == 1 and isinstance(segmentation[0], list):
            return normalize_polygon(segmentation[0])
    if isinstance(segmentation, list) and len(segmentation) % 2 == 0:
        return [(float(segmentation[idx]), float(segmentation[idx + 1])) for idx in range(0, len(segmentation), 2)]
    raise ValueError(f"Unsupported polygon format: {segmentation!r}")


def polygon_to_coco_coords(points: Sequence[tuple[float, float]]) -> list[float]:
    flattened = []
    for x_coord, y_coord in points:
        flattened.extend([float(x_coord), float(y_coord)])
    return flattened


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def bbox_from_polygons(polygons: Sequence[Sequence[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for polygon in polygons:
        xs.extend(point[0] for point in polygon)
        ys.extend(point[1] for point in polygon)
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return (float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y))


def dhash_image(image_path: Path, hash_size: int = 8) -> int:
    image = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size), resample=Image.BILINEAR)
    pixels = list(image.getdata())
    bits = 0
    bit_index = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            if left > right:
                bits |= 1 << bit_index
            bit_index += 1
    return bits


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def cluster_similar_images(image_paths: Sequence[Path], hamming_threshold: int = 6) -> list[list[int]]:
    hashes = [dhash_image(path) for path in image_paths]
    union_find = UnionFind(range(len(image_paths)))
    for left_idx in range(len(image_paths)):
        for right_idx in range(left_idx + 1, len(image_paths)):
            if hamming_distance(hashes[left_idx], hashes[right_idx]) <= hamming_threshold:
                union_find.union(left_idx, right_idx)
    groups = defaultdict(list)
    for idx in range(len(image_paths)):
        groups[union_find.find(idx)].append(idx)
    return list(groups.values())


def assign_split_by_groups(
    samples: Sequence[dict[str, Path]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    hamming_threshold: int = 6,
) -> dict[str, list[dict[str, Path]]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    image_paths = [Path(sample["image_path"]) for sample in samples]
    index_groups = cluster_similar_images(image_paths, hamming_threshold=hamming_threshold)
    random.Random(seed).shuffle(index_groups)
    split_targets = {
        "train": train_ratio * len(samples),
        "val": val_ratio * len(samples),
        "test": test_ratio * len(samples),
    }
    split_to_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_counts = {key: 0 for key in split_to_indices}

    for group in sorted(index_groups, key=len, reverse=True):
        target_split = min(split_counts, key=lambda name: split_counts[name] / max(split_targets[name], 1e-6))
        split_to_indices[target_split].extend(group)
        split_counts[target_split] += len(group)

    split_to_samples = {
        split: [samples[idx] for idx in sorted(indices)]
        for split, indices in split_to_indices.items()
    }
    return split_to_samples


def group_isat_instances(sample: dict[str, Path]) -> tuple[dict, list[GroupedInstance]]:
    annotation = load_isat_json(sample["annotation_path"])
    info = annotation["info"]
    width = int(info["width"])
    height = int(info["height"])

    grouped: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    grouped_points: dict[tuple[int, int], list[list[tuple[float, float]]]] = defaultdict(list)
    for obj in annotation.get("objects", []):
        category_name = canonical_category_name(obj.get("category", ""))
        category_id = CATEGORY_NAME_TO_ID[category_name]
        group = int(obj.get("group", 0))
        polygon_points = normalize_polygon(obj.get("segmentation"))
        if len(polygon_points) < 3:
            continue
        grouped[(category_id, group)].append(polygon_to_coco_coords(polygon_points))
        grouped_points[(category_id, group)].append(list(polygon_points))

    instances = []
    for (category_id, group), polygons in sorted(grouped.items()):
        point_polygons = grouped_points[(category_id, group)]
        area = sum(polygon_area(points) for points in point_polygons)
        bbox = bbox_from_polygons(point_polygons)
        category_name = "wire" if category_id == 1 else "interface-hole"
        instances.append(
            GroupedInstance(
                image_stem=sample["stem"],
                image_path=sample["image_path"],
                annotation_path=sample["annotation_path"],
                width=width,
                height=height,
                category_id=category_id,
                category_name=category_name,
                group=group,
                polygons=tuple(polygons),
                area=float(area),
                bbox=bbox,
                polygon_count=len(polygons),
            )
        )
    return annotation, instances


def ensure_clean_destination(dst_root: Path, overwrite: bool = False) -> None:
    if dst_root.exists() and any(dst_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination directory {dst_root} already exists and is not empty. Use --overwrite to continue."
        )
    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    (dst_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dst_root / "reviews").mkdir(parents=True, exist_ok=True)

