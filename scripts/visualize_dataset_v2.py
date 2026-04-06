"""Offline dataset visualizer for the S02 label/hole instance pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.label_hole_instance_dataset import LabelHoleInstanceDataset

CLASS_COLORS = {
    1: (70, 220, 120),
    2: (255, 120, 72),
}
CLASS_NAMES = {
    1: "label_sleeve",
    2: "empty_terminal",
}


def _overlay_masks(image: Image.Image, masks: list[np.ndarray], labels: list[int], alpha: float = 0.38) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for mask, label in zip(masks, labels):
        color = CLASS_COLORS.get(int(label), (255, 255, 255))
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        color_layer = Image.new("RGBA", base.size, color + (0,))
        color_layer.putalpha(mask_image.point(lambda value: int(round(value * alpha))))
        overlay = Image.alpha_composite(overlay, color_layer)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _oriented_box_to_polygon(oriented_box: list[float] | np.ndarray) -> list[tuple[float, float]]:
    cx, cy, width, height, angle = [float(value) for value in oriented_box]
    angle_rad = np.deg2rad(angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    dx = width * 0.5
    dy = height * 0.5
    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy),
    ]
    polygon = []
    for x_off, y_off in corners:
        x = cx + x_off * cos_a - y_off * sin_a
        y = cy + x_off * sin_a + y_off * cos_a
        polygon.append((float(x), float(y)))
    return polygon


def _principal_axis_segment(oriented_box: list[float] | np.ndarray, principal_axis: list[float] | np.ndarray) -> list[tuple[float, float]]:
    cx, cy, width, height, _ = [float(value) for value in oriented_box]
    axis_x, axis_y = [float(value) for value in principal_axis]
    scale = 0.5 * max(width, height, 1.0)
    return [
        (float(cx - axis_x * scale), float(cy - axis_y * scale)),
        (float(cx + axis_x * scale), float(cy + axis_y * scale)),
    ]


def _draw_geometry(
    image: Image.Image,
    boxes: list[list[float]],
    oriented_boxes: list[list[float]],
    principal_axes: list[list[float]],
    labels: list[int],
) -> Image.Image:
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    for box, oriented_box, principal_axis, label in zip(boxes, oriented_boxes, principal_axes, labels):
        color = CLASS_COLORS.get(int(label), (255, 255, 255))
        draw.rectangle(box, outline=color, width=3)
        draw.polygon(_oriented_box_to_polygon(oriented_box), outline=color)
        axis_segment = _principal_axis_segment(oriented_box, principal_axis)
        draw.line(axis_segment, fill=(255, 255, 255), width=2)
        for point in axis_segment:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(255, 255, 255))
    return result


def render_processed_overlay(sample: dict[str, Any]) -> Image.Image:
    image = Image.fromarray((sample["image"].mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()))
    instances = sample["instances"]
    masks = [mask.cpu().numpy() for mask in instances["masks"]]
    labels = [int(value) for value in instances["labels"].tolist()]
    boxes = [box.tolist() for box in instances["boxes"]]
    oriented_boxes = [box.tolist() for box in instances["oriented_boxes"]]
    principal_axes = [axis.tolist() for axis in instances["principal_axes"]]
    overlay = _overlay_masks(image, masks, labels)
    overlay = _draw_geometry(overlay, boxes, oriented_boxes, principal_axes, labels)
    return overlay


def render_raw_overlay(dataset: LabelHoleInstanceDataset, sample: dict[str, Any]) -> Image.Image:
    image = Image.open(sample["image_path"]).convert("RGB")
    raw_instances = dataset.load_original_instances(sample["image_id"])
    masks = [instance["mask"] for instance in raw_instances]
    labels = [int(instance["label"]) for instance in raw_instances]
    boxes = [instance["bbox_xyxy"].tolist() for instance in raw_instances]
    oriented_boxes = [instance["oriented_box"].tolist() for instance in raw_instances]
    principal_axes = [instance["principal_axis"].tolist() for instance in raw_instances]
    overlay = _overlay_masks(image, masks, labels)
    overlay = _draw_geometry(overlay, boxes, oriented_boxes, principal_axes, labels)
    crop_box = sample.get("crop_box")
    if crop_box is not None:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(crop_box, outline=(255, 255, 255), width=3)
    return overlay


def export_dataset_visualizations(
    dataset: LabelHoleInstanceDataset,
    output_dir: str | Path,
    limit: int = 8,
) -> list[tuple[Path, Path]]:
    output_root = Path(output_dir)
    raw_dir = output_root / "raw_overlay"
    crop_dir = output_root / "crop_overlay"
    raw_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    exported: list[tuple[Path, Path]] = []
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        image_id = int(sample["image_id"])
        raw_path = raw_dir / f"{index:03d}_image_{image_id:05d}.png"
        crop_path = crop_dir / f"{index:03d}_image_{image_id:05d}.png"
        render_raw_overlay(dataset, sample).save(raw_path)
        render_processed_overlay(sample).save(crop_path)
        exported.append((raw_path, crop_path))
    return exported


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="samDataset_instance_coco")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="visualizations/dataset_v2")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-image-prob", type=float, default=0.5)
    parser.add_argument("--object-crop-prob", type=float, default=0.5)
    parser.add_argument("--hole-focused-prob", type=float, default=0.6)
    parser.add_argument("--label-focused-prob", type=float, default=0.4)
    parser.add_argument("--no-augment", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset = LabelHoleInstanceDataset(
        data_root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        full_image_prob=args.full_image_prob,
        object_crop_prob=args.object_crop_prob,
        hole_focused_prob=args.hole_focused_prob,
        label_focused_prob=args.label_focused_prob,
        seed=args.seed,
        augment=not args.no_augment,
    )
    export_dataset_visualizations(dataset, args.output_dir, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
