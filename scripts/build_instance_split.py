#!/usr/bin/env python3
"""
Build a grouped split manifest for a flat ISAT dataset without exporting COCO JSON.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.coco_export import assign_split_by_groups, discover_isat_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped train/val/test split manifest for ISAT data.")
    parser.add_argument("--src", required=True, help="Source flat ISAT dataset directory.")
    parser.add_argument("--dst", required=True, help="Destination CSV manifest path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--hash-threshold", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = discover_isat_samples(Path(args.src).expanduser().resolve())
    split_to_samples = assign_split_by_groups(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        hamming_threshold=args.hash_threshold,
    )
    dst_path = Path(args.dst).expanduser().resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "base", "image", "annotation"])
        writer.writeheader()
        for split, split_samples in split_to_samples.items():
            for sample in split_samples:
                writer.writerow(
                    {
                        "split": split,
                        "base": sample["stem"],
                        "image": sample["image_path"].name,
                        "annotation": sample["annotation_path"].name,
                    }
                )
    print(f"Wrote grouped split manifest to {dst_path}")


if __name__ == "__main__":
    main()
