#!/usr/bin/env python3
"""Convert ISAT annotations into the WireCR-HQInstSAM COCO v2 format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.geometry_utils import (
    bbox_xywh_from_mask,
    flatten_polygon,
    load_isat_json,
    multi_polygon_to_mask,
    normalize_polygon,
    oriented_box_from_points,
    oriented_box_from_mask,
    polygon_raster_area,
    principal_axis_from_mask,
)

CATEGORIES = [
    {"id": 1, "name": "label_sleeve", "supercategory": "foreground"},
    {"id": 2, "name": "empty_terminal", "supercategory": "foreground"},
]

CATEGORY_NAME_TO_ID = {
    "wire": 1,
    "interface-hole": 2,
}


def normalize_category(name: str) -> str:
    normalized = str(name).strip()
    if normalized == "wire":
        return "label_sleeve"
    if normalized == "interface-hole":
        return "empty_terminal"
    raise ValueError(f"Unsupported category {name!r}. Expected only wire/interface-hole.")


def _category_id_from_name(name: str) -> int:
    return 1 if name == "label_sleeve" else 2


def discover_samples(src_root: Path) -> list[dict[str, Path]]:
    manifest_path = src_root / "labeled_manifest.csv"
    if manifest_path.is_file():
        samples: list[dict[str, Path]] = []
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                samples.append(
                    {
                        "stem": row["base"],
                        "image_path": src_root / row["image"],
                        "annotation_path": src_root / row["annotation"],
                    }
                )
        return samples

    samples = []
    for annotation_path in sorted(src_root.glob("*.json")):
        stem = annotation_path.stem
        image_path = None
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = src_root / f"{stem}{suffix}"
            if candidate.is_file():
                image_path = candidate
                break
        if image_path is None:
            raise FileNotFoundError(f"Could not find image for annotation {annotation_path.name}")
        samples.append({"stem": stem, "image_path": image_path, "annotation_path": annotation_path})
    if not samples:
        raise FileNotFoundError(f"No ISAT JSON annotations found under {src_root}")
    return samples


def group_objects_to_instances(
    objects: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    min_instance_area: float = 0.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for obj in objects:
        category_name = normalize_category(obj.get("category", ""))
        group_id = int(obj.get("group", 0))
        polygon = normalize_polygon(obj.get("segmentation"))
        if len(polygon) < 3:
            continue
        key = (_category_id_from_name(category_name), group_id)
        bucket = grouped.setdefault(
            key,
            {
                "category_id": key[0],
                "category_name": category_name,
                "group_id": group_id,
                "polygons": [],
                "width": width,
                "height": height,
            },
        )
        bucket["polygons"].append(polygon)

    instances: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        polygons = bucket["polygons"]
        mask = multi_polygon_to_mask(int(bucket["height"]), int(bucket["width"]), polygons)
        area = float(sum(sum(row) for row in mask))
        if area < float(min_instance_area):
            continue
        instances.append(
            {
                "category_id": bucket["category_id"],
                "category_name": bucket["category_name"],
                "group_id": bucket["group_id"],
                "polygons": polygons,
                "bbox": bbox_xywh_from_mask(mask),
                "oriented_box": oriented_box_from_mask(mask),
                "principal_axis": principal_axis_from_mask(mask),
                "area": area,
                "mask": mask,
                "polygon_count": len(polygons),
            }
        )
    return instances


def compute_oriented_box(mask_or_polygon: Any) -> list[float]:
    if hasattr(mask_or_polygon, "__iter__") and mask_or_polygon and isinstance(mask_or_polygon[0], (list, tuple)):
        if mask_or_polygon and mask_or_polygon[0] and isinstance(mask_or_polygon[0][0], (int, float)):
            return oriented_box_from_points(normalize_polygon(mask_or_polygon))
        if mask_or_polygon and mask_or_polygon[0] and isinstance(mask_or_polygon[0][0], (list, tuple)):
            points = [point for polygon in mask_or_polygon for point in normalize_polygon(polygon)]
            return oriented_box_from_points(points)
    if isinstance(mask_or_polygon, list) and not mask_or_polygon:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    raise TypeError(f"Unsupported input for compute_oriented_box: {type(mask_or_polygon).__name__}")


def _random_split(samples: list[dict[str, Path]], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[dict[str, Path]]]:
    train_ratio, val_ratio, test_ratio = ratios
    total = len(samples)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    train_end = int(round(total * train_ratio))
    val_end = train_end + int(round(total * val_ratio))
    splits = {
        "train": [samples[idx] for idx in indices[:train_end]],
        "val": [samples[idx] for idx in indices[train_end:val_end]],
        "test": [samples[idx] for idx in indices[val_end:]],
    }
    return splits


def _dhash_image(image_path: Path, hash_size: int = 8) -> int:
    from PIL import Image

    image = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size), resample=Image.BILINEAR)
    pixels = list(image.getdata())
    bits = 0
    bit_index = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            if pixels[row_offset + col] > pixels[row_offset + col + 1]:
                bits |= 1 << bit_index
            bit_index += 1
    return bits


def _hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def _dedupe_groups(samples: list[dict[str, Path]], dedupe_threshold: int) -> list[list[int]]:
    hashes = [_dhash_image(sample["image_path"]) for sample in samples]
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        if parent[index] != index:
            parent[index] = find(parent[index])
        return parent[index]

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if _hamming_distance(hashes[left], hashes[right]) <= int(dedupe_threshold):
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        groups[find(index)].append(index)
    return list(groups.values())


def split_dataset(
    samples: list[dict[str, Path]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
    split_mode: str = "dedupe",
    dedupe_threshold: int = 6,
) -> dict[str, list[dict[str, Path]]]:
    train_ratio, val_ratio, test_ratio = ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    if split_mode == "random":
        return _random_split(samples, seed=seed, ratios=ratios)

    groups = _dedupe_groups(samples, dedupe_threshold=dedupe_threshold)
    random.Random(seed).shuffle(groups)
    split_targets = {
        "train": train_ratio * len(samples),
        "val": val_ratio * len(samples),
        "test": test_ratio * len(samples),
    }
    split_to_indices = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    for group in sorted(groups, key=len, reverse=True):
        target = min(counts, key=lambda name: counts[name] / max(split_targets[name], 1e-6))
        split_to_indices[target].extend(group)
        counts[target] += len(group)

    return {
        split: [samples[index] for index in sorted(indices)]
        for split, indices in split_to_indices.items()
    }


def load_isat_sample(annotation_path: Path) -> dict[str, Any]:
    annotation = load_isat_json(annotation_path)
    if "info" not in annotation or "objects" not in annotation:
        raise KeyError(f"Malformed ISAT file {annotation_path}")
    return annotation


def export_coco(
    *,
    src_root: Path,
    dst_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    dedupe_threshold: int,
    split_mode: str,
    min_instance_area: float,
    copy_images: bool,
) -> dict[str, Any]:
    samples = discover_samples(src_root)
    split_to_samples = split_dataset(
        samples,
        seed=seed,
        ratios=(train_ratio, val_ratio, test_ratio),
        split_mode=split_mode,
        dedupe_threshold=dedupe_threshold,
    )

    if dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(f"Destination {dst_root} already exists and is not empty. Use --overwrite.")
    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    (dst_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dst_root / "reviews").mkdir(parents=True, exist_ok=True)

    split_manifest_rows = []
    review_rows = []
    report = {
        "split_mode": split_mode,
        "dedupe_threshold": int(dedupe_threshold),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "images": {},
        "annotations": {},
        "classes": {},
        "exceptions": [],
    }
    global_image_id = 1
    global_annotation_id = 1

    for split, split_samples in split_to_samples.items():
        image_dir = dst_root / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        class_counts: Counter[str] = Counter()
        instance_area_total = 0.0
        instance_count = 0
        for sample in split_samples:
            annotation = load_isat_sample(sample["annotation_path"])
            info = annotation["info"]
            width = int(info["width"])
            height = int(info["height"])
            objects = annotation["objects"]
            grouped: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"polygons": []})
            for obj in objects:
                category_name = normalize_category(obj.get("category", ""))
                category_id = 1 if category_name == "label_sleeve" else 2
                group_id = int(obj.get("group", 0))
                polygon = normalize_polygon(obj.get("segmentation"))
                if len(polygon) < 3:
                    continue
                grouped[(category_id, group_id)]["category_name"] = category_name
                grouped[(category_id, group_id)]["group_id"] = group_id
                grouped[(category_id, group_id)]["category_id"] = category_id
                grouped[(category_id, group_id)]["polygons"].append(polygon)

            images.append(
                {
                    "id": global_image_id,
                    "file_name": sample["image_path"].name,
                    "width": width,
                    "height": height,
                }
            )

            for (category_id, group_id), bucket in sorted(grouped.items()):
                polygons = bucket["polygons"]
                mask = multi_polygon_to_mask(height, width, polygons)
                area = float(sum(sum(row) for row in mask))
                if area < float(min_instance_area):
                    continue
                class_name = "label_sleeve" if category_id == 1 else "empty_terminal"
                bbox = bbox_xywh_from_mask(mask)
                oriented_box = oriented_box_from_mask(mask)
                principal_axis = principal_axis_from_mask(mask)
                annotations.append(
                    {
                        "id": global_annotation_id,
                        "image_id": global_image_id,
                        "category_id": category_id,
                        "segmentation": [flatten_polygon(polygon) for polygon in polygons],
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "group_id": group_id,
                        "oriented_box": oriented_box,
                        "principal_axis": principal_axis,
                    }
                )
                review_rows.append(
                    {
                        "split": split,
                        "image_id": global_image_id,
                        "image_name": sample["image_path"].name,
                        "category": class_name,
                        "group_id": group_id,
                        "polygon_count": len(polygons),
                        "bbox": " ".join(f"{value:.2f}" for value in bbox),
                        "area": f"{area:.2f}",
                    }
                )
                class_counts[class_name] += 1
                instance_area_total += area
                instance_count += 1
                global_annotation_id += 1

            if copy_images:
                shutil.copy2(sample["image_path"], image_dir / sample["image_path"].name)
            else:
                destination = image_dir / sample["image_path"].name
                if destination.exists():
                    destination.unlink()
                try:
                    os.symlink(sample["image_path"], destination)
                except OSError:
                    shutil.copy2(sample["image_path"], destination)

            split_manifest_rows.append(
                {
                    "split": split,
                    "base": sample["stem"],
                    "image": f"images/{split}/{sample['image_path'].name}",
                    "annotation": sample["annotation_path"].name,
                }
            )
            global_image_id += 1

        coco_json = {"images": images, "annotations": annotations, "categories": CATEGORIES}
        with (dst_root / "annotations" / f"instances_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(coco_json, handle, ensure_ascii=False, indent=2)

        report["images"][split] = len(images)
        report["annotations"][split] = len(annotations)
        report["classes"][split] = dict(class_counts)
        if annotations:
            report[f"{split}_avg_area"] = round(instance_area_total / max(instance_count, 1), 4)

    with (dst_root / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "base", "image", "annotation"])
        writer.writeheader()
        writer.writerows(split_manifest_rows)

    with (dst_root / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with (dst_root / "reviews" / "instance_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "image_id", "image_name", "category", "group_id", "polygon_count", "bbox", "area"],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert flat ISAT data into COCO v2 for WireCR-HQInstSAM.")
    parser.add_argument("--src", required=True, help="Source ISAT dataset directory.")
    parser.add_argument("--dst", required=True, help="Destination COCO v2 dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the destination directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--min-instance-area", type=float, default=0.0, help="Drop instances below this area.")
    parser.add_argument("--dedupe-threshold", type=int, default=6, help="Dedupe hamming threshold.")
    parser.add_argument(
        "--split-mode",
        choices=["dedupe", "random"],
        default="dedupe",
        help="Split strategy. Default is dedupe; random is only for smoke tests.",
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy images into the COCO output tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    if dst_root.exists() and any(dst_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Destination directory {dst_root} already exists and is not empty. Use --overwrite.")
    if args.overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    report = export_coco(
        src_root=src_root,
        dst_root=dst_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dedupe_threshold=args.dedupe_threshold,
        split_mode=args.split_mode,
        min_instance_area=args.min_instance_area,
        copy_images=bool(args.copy_images),
    )

    print(f"Converted {sum(report['images'].values())} images into {dst_root}")
    print(f"Split mode: {report['split_mode']}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {report['images'].get(split, 0)} images, {report['annotations'].get(split, 0)} annotations")
    print(f"Conversion report: {dst_root / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
