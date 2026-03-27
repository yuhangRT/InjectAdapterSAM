#!/usr/bin/env python3
"""
Quick consistency report for ISAT group annotations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.coco_export import discover_isat_samples, group_isat_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check grouped instance consistency in flat ISAT annotations.")
    parser.add_argument("--src", required=True, help="Source flat ISAT dataset directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = discover_isat_samples(Path(args.src).expanduser().resolve())
    category_counts = Counter()
    polygon_counts = Counter()
    per_image = []

    for sample in samples:
        _, instances = group_isat_instances(sample)
        grouped_per_category = defaultdict(int)
        for instance in instances:
            category_counts[instance.category_name] += 1
            polygon_counts[instance.category_name] += instance.polygon_count
            grouped_per_category[instance.category_name] += 1
        per_image.append((sample["stem"], dict(grouped_per_category)))

    print("Category instance counts:")
    for category_name in sorted(category_counts):
        print(
            f"  {category_name}: instances={category_counts[category_name]}, "
            f"polygons={polygon_counts[category_name]}"
        )
    print("\nPer-image preview:")
    for stem, counts in per_image[:10]:
        print(f"  {stem}: {counts}")


if __name__ == "__main__":
    main()
