#!/usr/bin/env python3
"""
Convert flat ISAT annotations into a COCO-style instance segmentation dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.coco_export import (
    CATEGORIES,
    assign_split_by_groups,
    discover_isat_samples,
    ensure_clean_destination,
    group_isat_instances,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ISAT polygons to COCO instance annotations.")
    parser.add_argument("--src", required=True, help="Source flat ISAT dataset directory.")
    parser.add_argument("--dst", required=True, help="Destination COCO-style dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for split assignment.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--hash-threshold", type=int, default=6, help="Hamming threshold used for near-duplicate grouping.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it already exists.")
    return parser.parse_args()


def build_coco_json(split: str, samples: list[dict[str, Path]]) -> tuple[dict, list[dict[str, object]]]:
    images = []
    annotations = []
    review_rows = []
    annotation_id = 1

    for image_id, sample in enumerate(samples, start=1):
        annotation, instances = group_isat_instances(sample)
        info = annotation["info"]
        images.append(
            {
                "id": image_id,
                "file_name": sample["image_path"].name,
                "width": int(info["width"]),
                "height": int(info["height"]),
            }
        )
        for instance in instances:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": instance.category_id,
                    "segmentation": [list(polygon) for polygon in instance.polygons],
                    "bbox": list(instance.bbox),
                    "area": float(instance.area),
                    "iscrowd": 0,
                    "group": int(instance.group),
                }
            )
            review_rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "image_name": sample["image_path"].name,
                    "category": instance.category_name,
                    "group": int(instance.group),
                    "polygon_count": int(instance.polygon_count),
                    "bbox": " ".join(f"{value:.2f}" for value in instance.bbox),
                    "area": f"{instance.area:.2f}",
                }
            )
            annotation_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    return coco, review_rows


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    ensure_clean_destination(dst_root, overwrite=args.overwrite)

    samples = discover_isat_samples(src_root)
    split_to_samples = assign_split_by_groups(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        hamming_threshold=args.hash_threshold,
    )

    split_manifest_path = dst_root / "split_manifest.csv"
    review_rows = []
    with split_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "base", "image", "annotation"])
        writer.writeheader()
        for split, split_samples in split_to_samples.items():
            image_dir = dst_root / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            coco_json, split_review_rows = build_coco_json(split, split_samples)
            review_rows.extend(split_review_rows)

            annotation_path = dst_root / "annotations" / f"instances_{split}.json"
            with annotation_path.open("w", encoding="utf-8") as coco_handle:
                json.dump(coco_json, coco_handle, ensure_ascii=False, indent=2)

            for sample in split_samples:
                shutil.copy2(sample["image_path"], image_dir / sample["image_path"].name)
                writer.writerow(
                    {
                        "split": split,
                        "base": sample["stem"],
                        "image": f"images/{split}/{sample['image_path'].name}",
                        "annotation": sample["annotation_path"].name,
                    }
                )

    review_path = dst_root / "reviews" / "instance_review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "image_id", "image_name", "category", "group", "polygon_count", "bbox", "area"],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"Converted {len(samples)} images into {dst_root}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(split_to_samples[split])}")
    print(f"Review CSV: {review_path}")


if __name__ == "__main__":
    main()
