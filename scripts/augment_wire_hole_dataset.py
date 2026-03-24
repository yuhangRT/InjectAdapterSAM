#!/usr/bin/env python3
"""
Expand a wire_hole dataset with offline industrial augmentations on the train split.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.industrial_augment import build_train_augmentor


MASK_PALETTE = [
    0, 0, 0,
    0, 255, 0,
    255, 0, 0,
] + [0, 0, 0] * 253
MASK_RGB_TO_LABEL = {
    (0, 0, 0): 0,
    (0, 255, 0): 1,
    (255, 0, 0): 2,
}
MASK_LABEL_ALIASES = {
    0: 0,
    1: 1,
    2: 2,
    76: 1,
    150: 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline augmented wire_hole dataset by expanding the train split."
    )
    parser.add_argument("--src", required=True, help="Source wire_hole dataset root.")
    parser.add_argument("--dst", required=True, help="Destination dataset root.")
    parser.add_argument(
        "--train-copies",
        type=int,
        default=2,
        help="Number of augmented copies generated for each training sample.",
    )
    parser.add_argument(
        "--augment-strength",
        type=str,
        default="medium",
        choices=["light", "medium", "strong"],
        help="Industrial augmentation preset strength.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used to generate deterministic augmented copies.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty destination directory.",
    )
    return parser.parse_args()


def _resolve_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find split directories for '{split}' under {root}. Expected images/{split} and masks/{split}."
        )
    return image_dir, mask_dir


def _prepare_destination(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists() and any(dst_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination directory {dst_root} already exists and is not empty. Use --overwrite to continue."
        )
    if dst_root.exists() and overwrite:
        for rel_path in ("images", "masks"):
            path = dst_root / rel_path
            if path.exists():
                shutil.rmtree(path)
        manifest_path = dst_root / "split_manifest.csv"
        if manifest_path.exists():
            manifest_path.unlink()

    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "masks" / split).mkdir(parents=True, exist_ok=True)


def _collect_samples(root: Path, split: str) -> list[dict[str, Path | str]]:
    image_dir, mask_dir = _resolve_split_dirs(root, split)
    mask_index = {path.stem: path for path in sorted(mask_dir.iterdir()) if path.is_file()}

    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        mask_path = mask_index.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for image: {image_path}")
        samples.append(
            {
                "split": split,
                "stem": image_path.stem,
                "image_path": image_path,
                "mask_path": mask_path,
                "annotation": f"{image_path.stem}.json",
            }
        )
    return samples


def _copy_sample(sample: dict[str, Path | str], dst_root: Path, stem: str) -> dict[str, str]:
    split = str(sample["split"])
    src_image = Path(sample["image_path"])
    src_mask = Path(sample["mask_path"])

    dst_image = dst_root / "images" / split / f"{stem}{src_image.suffix.lower()}"
    dst_mask = dst_root / "masks" / split / f"{stem}.png"

    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_mask, dst_mask)

    return {
        "split": split,
        "base": stem,
        "image": f"images/{split}/{dst_image.name}",
        "mask": f"masks/{split}/{dst_mask.name}",
        "annotation": str(sample["annotation"]),
    }


def _mask_to_class_ids(mask: Image.Image) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        if mask_array.shape[-1] == 1:
            mask_array = mask_array[..., 0]
        elif mask_array.shape[-1] == 3:
            flat = mask_array.reshape(-1, 3)
            unique_colors = {
                tuple(int(channel) for channel in color.tolist())
                for color in np.unique(flat, axis=0)
            }
            unknown_colors = sorted(unique_colors.difference(MASK_RGB_TO_LABEL))
            if unknown_colors:
                raise ValueError(
                    f"Unsupported augmented mask colors {unknown_colors}. "
                    f"Expected only {sorted(MASK_RGB_TO_LABEL)}."
                )
            remapped = np.zeros(mask_array.shape[:2], dtype=np.uint8)
            for color, class_idx in MASK_RGB_TO_LABEL.items():
                remapped[np.all(mask_array == np.asarray(color, dtype=mask_array.dtype), axis=-1)] = class_idx
            return remapped
        else:
            raise ValueError(f"Unsupported augmented mask shape {mask_array.shape}.")

    unique_values = {int(value) for value in np.unique(mask_array).tolist()}
    unknown_values = sorted(unique_values.difference(MASK_LABEL_ALIASES))
    if unknown_values:
        raise ValueError(
            f"Unsupported augmented mask labels {unknown_values}. "
            f"Expected only {sorted(MASK_LABEL_ALIASES)}."
        )

    remapped = mask_array.astype(np.uint8).copy()
    for raw_value, class_idx in MASK_LABEL_ALIASES.items():
        remapped[mask_array == raw_value] = class_idx
    return remapped


def _save_augmented_mask(mask: Image.Image, dst_path: Path) -> None:
    class_ids = _mask_to_class_ids(mask)
    palette_mask = Image.fromarray(class_ids, mode="P")
    palette_mask.putpalette(MASK_PALETTE)
    palette_mask.save(dst_path)


def _augment_train_sample(
    sample: dict[str, Path | str],
    *,
    dst_root: Path,
    augmentor,
    train_copies: int,
    seed: int,
) -> list[dict[str, str]]:
    split = str(sample["split"])
    image_path = Path(sample["image_path"])
    mask_path = Path(sample["mask_path"])
    stem = str(sample["stem"])

    generated_rows = []
    for copy_idx in range(1, train_copies + 1):
        stem_value = sum((idx + 1) * ord(char) for idx, char in enumerate(stem))
        rng_seed = (seed + stem_value * 1009 + copy_idx * 9173) % (2**32)
        rng = np.random.default_rng(rng_seed)

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            aug_image, aug_mask = augmentor(image, mask, rng)

        aug_stem = f"{stem}__aug{copy_idx:02d}"
        dst_image = dst_root / "images" / split / f"{aug_stem}{image_path.suffix.lower()}"
        dst_mask = dst_root / "masks" / split / f"{aug_stem}.png"

        aug_image.save(dst_image)
        _save_augmented_mask(aug_mask, dst_mask)

        generated_rows.append(
            {
                "split": split,
                "base": aug_stem,
                "image": f"images/{split}/{dst_image.name}",
                "mask": f"masks/{split}/{dst_mask.name}",
                "annotation": str(sample["annotation"]),
            }
        )

    return generated_rows


def _write_manifest(dst_root: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = dst_root / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "base", "image", "mask", "annotation"])
        writer.writeheader()
        writer.writerows(rows)


def augment_dataset(args: argparse.Namespace) -> dict[str, int]:
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")
    if args.train_copies < 0:
        raise ValueError(f"--train-copies must be >= 0, got {args.train_copies}")

    augmentor = build_train_augmentor("industrial", args.augment_strength, offline_mode=True)
    _prepare_destination(dst_root, overwrite=args.overwrite)

    manifest_rows: list[dict[str, str]] = []
    counts = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        samples = _collect_samples(src_root, split)
        for sample in samples:
            row = _copy_sample(sample, dst_root, stem=str(sample["stem"]))
            manifest_rows.append(row)
            counts[split] += 1

            if split == "train" and args.train_copies > 0:
                aug_rows = _augment_train_sample(
                    sample,
                    dst_root=dst_root,
                    augmentor=augmentor,
                    train_copies=args.train_copies,
                    seed=args.seed,
                )
                manifest_rows.extend(aug_rows)
                counts[split] += len(aug_rows)

    _write_manifest(dst_root, manifest_rows)
    return counts


def main() -> None:
    args = parse_args()
    counts = augment_dataset(args)
    total = sum(counts.values())
    print(f"Created offline augmented dataset at {Path(args.dst).resolve()}")
    print(f"  total: {total}")
    print(f"  train: {counts['train']}")
    print(f"  val: {counts['val']}")
    print(f"  test: {counts['test']}")


if __name__ == "__main__":
    main()
