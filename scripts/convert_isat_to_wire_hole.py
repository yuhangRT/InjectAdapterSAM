#!/usr/bin/env python3
"""
Convert a flat ISAT polygon dataset into the wire_hole directory layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CLASS_NAME_TO_LABEL = {
    "__background__": 0,
    "background": 0,
    "wire": 1,
    "hole": 2,
    "interface-hole": 2,
}
DRAW_ORDER = ("wire", "interface-hole")
MASK_PALETTE = [
    0,
    0,
    0,
    0,
    255,
    0,
    255,
    0,
    0,
] + [0, 0, 0] * 253


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ISAT JSON annotations into wire_hole masks.")
    parser.add_argument("--src", required=True, help="Source flat ISAT dataset directory.")
    parser.add_argument("--dst", required=True, help="Destination wire_hole dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for split shuffling.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty destination directory.",
    )
    return parser.parse_args()


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(f"All split ratios must be positive, got {ratios}.")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} -> {total}.")


def _resolve_image_path(src_root: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = src_root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find an image file for stem '{stem}' under {src_root}.")


def _load_samples_from_manifest(src_root: Path, manifest_path: Path) -> List[Dict[str, Path]]:
    samples = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stem = row["base"]
            image_path = src_root / row["image"]
            annotation_path = src_root / row["annotation"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in manifest does not exist: {image_path}")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Annotation listed in manifest does not exist: {annotation_path}")
            samples.append({"stem": stem, "image_path": image_path, "annotation_path": annotation_path})
    return samples


def _discover_samples(src_root: Path) -> List[Dict[str, Path]]:
    manifest_path = src_root / "labeled_manifest.csv"
    if manifest_path.is_file():
        return _load_samples_from_manifest(src_root, manifest_path)

    samples = []
    for annotation_path in sorted(src_root.glob("*.json")):
        stem = annotation_path.stem
        image_path = _resolve_image_path(src_root, stem)
        samples.append({"stem": stem, "image_path": image_path, "annotation_path": annotation_path})
    if not samples:
        raise FileNotFoundError(f"No ISAT JSON annotations found under {src_root}.")
    return samples


def _split_samples(
    samples: Sequence[Dict[str, Path]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[Dict[str, Path]]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    train_end = int(train_ratio * len(shuffled))
    val_end = int((train_ratio + val_ratio) * len(shuffled))
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if "info" not in data:
        raise ValueError(f"Missing 'info' field in annotation: {annotation_path}")
    return data


def _normalize_polygon(segmentation: object) -> List[Tuple[float, float]]:
    if not segmentation:
        return []
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        if len(segmentation[0]) == 2:
            return [(float(x), float(y)) for x, y in segmentation]
        if len(segmentation) == 1 and isinstance(segmentation[0], list):
            return _normalize_polygon(segmentation[0])
    if isinstance(segmentation, list) and len(segmentation) % 2 == 0:
        points = []
        for idx in range(0, len(segmentation), 2):
            points.append((float(segmentation[idx]), float(segmentation[idx + 1])))
        return points
    raise ValueError(f"Unsupported polygon format: {segmentation!r}")


def _build_mask(annotation: dict, annotation_path: Path) -> Image.Image:
    info = annotation["info"]
    width = int(info["width"])
    height = int(info["height"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    objects = annotation.get("objects", [])
    for category in DRAW_ORDER:
        label = CLASS_NAME_TO_LABEL[category]
        for obj in objects:
            raw_category = str(obj.get("category", "")).strip()
            canonical_category = "interface-hole" if CLASS_NAME_TO_LABEL.get(raw_category) == 2 else raw_category
            if raw_category not in CLASS_NAME_TO_LABEL:
                raise ValueError(
                    f"Unsupported category '{raw_category}' in {annotation_path}. "
                    f"Expected one of {sorted(CLASS_NAME_TO_LABEL)}."
                )
            if canonical_category != category:
                continue
            polygon = _normalize_polygon(obj.get("segmentation"))
            if len(polygon) < 3:
                continue
            draw.polygon(polygon, fill=label)

    return mask


def _apply_mask_palette(mask: Image.Image) -> Image.Image:
    palette_mask = mask.convert("P")
    palette_mask.putpalette(MASK_PALETTE)
    return palette_mask


def _prepare_destination(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists() and any(dst_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination directory {dst_root} already exists and is not empty. Use --overwrite to continue."
        )
    if dst_root.exists() and overwrite:
        for path in (dst_root / "images", dst_root / "masks"):
            if path.exists():
                shutil.rmtree(path)
        manifest_path = dst_root / "split_manifest.csv"
        if manifest_path.exists():
            manifest_path.unlink()
    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "masks" / split).mkdir(parents=True, exist_ok=True)


def _write_manifest(dst_root: Path, split_to_samples: Dict[str, Sequence[Dict[str, Path]]]) -> None:
    manifest_path = dst_root / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "base", "image", "mask", "annotation"])
        writer.writeheader()
        for split, samples in split_to_samples.items():
            for sample in samples:
                writer.writerow(
                    {
                        "split": split,
                        "base": sample["stem"],
                        "image": f"images/{split}/{sample['image_path'].name}",
                        "mask": f"masks/{split}/{sample['stem']}.png",
                        "annotation": str(sample["annotation_path"].name),
                    }
                )


def convert_dataset(args: argparse.Namespace) -> Dict[str, int]:
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")

    _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    samples = _discover_samples(src_root)
    split_to_samples = _split_samples(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    _prepare_destination(dst_root, overwrite=args.overwrite)

    for split, split_samples in split_to_samples.items():
        for sample in split_samples:
            annotation = _load_annotation(sample["annotation_path"])
            mask = _apply_mask_palette(_build_mask(annotation, sample["annotation_path"]))

            dst_image_path = dst_root / "images" / split / sample["image_path"].name
            dst_mask_path = dst_root / "masks" / split / f"{sample['stem']}.png"

            shutil.copy2(sample["image_path"], dst_image_path)
            mask.save(dst_mask_path)

    _write_manifest(dst_root, split_to_samples)
    return {split: len(split_samples) for split, split_samples in split_to_samples.items()}


def main() -> None:
    args = parse_args()
    counts = convert_dataset(args)
    total = sum(counts.values())
    print(f"Converted {total} labeled samples into {Path(args.dst).resolve()}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts.get(split, 0)}")


if __name__ == "__main__":
    main()
