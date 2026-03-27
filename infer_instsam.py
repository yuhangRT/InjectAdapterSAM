"""
Single-image / directory inference for WireCR-InstSAM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.init import init_device
from utils.instsam_cli import add_common_instsam_args
from utils.instsam_runtime import build_instsam_model, filter_predictions, load_checkpoint_into_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for WireCR-InstSAM.")
    add_common_instsam_args(parser)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str)
    input_group.add_argument("--image-dir", type=str)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=str, default="./inference_instsam")
    parser.add_argument("--wire-score-thresh", type=float, default=0.20)
    parser.add_argument("--hole-score-thresh", type=float, default=0.20)
    parser.add_argument("--wire-mask-nms-iou", type=float, default=0.60)
    parser.add_argument("--hole-mask-nms-iou", type=float, default=0.60)
    parser.add_argument("--topk-final", type=int, default=50)
    return parser.parse_args()


def discover_images(args) -> list[Path]:
    if args.image:
        return [Path(args.image).expanduser().resolve()]
    image_dir = Path(args.image_dir).expanduser().resolve()
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def preprocess_image(image_path: Path, image_size: int) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size[::-1]
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    scale_y = image_size / max(original_size[0], 1)
    scale_x = image_size / max(original_size[1], 1)
    return {
        "image": tensor,
        "original_size": original_size,
        "processed_size": (image_size, image_size),
        "full_image_size": original_size,
        "resize_scale": (scale_y, scale_x),
        "crop_box": None,
        "image_path": str(image_path),
    }


def scale_box_to_full(box: torch.Tensor, resize_scale: tuple[float, float]) -> list[float]:
    scale_y, scale_x = resize_scale
    scaled = box.detach().cpu().float().clone()
    scaled[[0, 2]] /= scale_x
    scaled[[1, 3]] /= scale_y
    return scaled.tolist()


def main() -> None:
    args = parse_args()
    device, _ = init_device(seed=args.seed, cpu=args.cpu, gpu=args.gpu, affinity=None)
    model = build_instsam_model(args, device)
    load_checkpoint_into_model(model, args.checkpoint)
    model.eval()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in discover_images(args):
        sample = preprocess_image(image_path, args.image_size)
        with torch.no_grad():
            images = sample["image"].unsqueeze(0).to(device)
            features = model.forward_backbone(images)
            proposal_outputs = model.forward_proposals(features)
            decoded = model.decode_proposals(proposal_outputs, image_size=args.image_size)
            refined = model.refine_instances(
                features=features,
                proposals=decoded,
                processed_sizes=[sample["processed_size"]],
                output_sizes=[sample["full_image_size"]],
                apply_roi_refiner=args.enable_roi_refiner,
            )
        predictions = []
        for proposal, refined_instance in zip(decoded[0], refined[0]["instances"]):
            predictions.append(
                {
                    "instance_id": refined_instance["instance_id"],
                    "category_id": refined_instance["category_id"],
                    "score": refined_instance["score"],
                    "bbox_processed": proposal["bbox"].detach().cpu(),
                    "bbox_full": torch.tensor(scale_box_to_full(proposal["bbox"], sample["resize_scale"])),
                    "mask_processed": refined_instance["mask_processed"].cpu(),
                    "mask_full": refined_instance["mask"].cpu(),
                    "source_prompt": refined_instance["source_prompt"],
                }
            )
        filtered = filter_predictions(
            [predictions],
            wire_threshold=args.wire_score_thresh,
            hole_threshold=args.hole_score_thresh,
            wire_nms_iou=args.wire_mask_nms_iou,
            hole_nms_iou=args.hole_mask_nms_iou,
            topk_per_class=args.topk_final,
        )[0]
        result = []
        for prediction in filtered:
            result.append(
                {
                    "instance_id": int(prediction["instance_id"]),
                    "category_id": int(prediction["category_id"]),
                    "score": float(prediction["score"]),
                    "bbox": [float(value) for value in prediction["bbox_full"].tolist()],
                    "source_prompt": prediction["source_prompt"],
                }
            )
        with (output_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
