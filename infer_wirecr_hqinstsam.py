"""Inference entrypoint for WireCR-HQInstSAM sliding-window prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from engine.inferencer import WireCRHQInstSAMInferencer
from models.wirecr_hq_instsam import WireCRHQInstSAM
from utils.checkpoint import load_checkpoint
from utils.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WireCR-HQInstSAM inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--image", default=None, help="Single image path.")
    parser.add_argument("--image-dir", default=None, help="Directory of images.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/PNG/masks/crops.")
    return parser


def _load_thresholds(checkpoint_path: str | Path, infer_cfg: dict[str, object]) -> dict[str, float]:
    state = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
    thresholds = dict(state.get("best_val_thresholds", {}))
    thresholds.setdefault("score_thresh_label", infer_cfg.get("score_thresh_label", 0.5))
    thresholds.setdefault("score_thresh_hole", infer_cfg.get("score_thresh_hole", 0.5))
    thresholds.setdefault("mask_nms_iou_label", infer_cfg.get("mask_nms_iou_label", 0.6))
    thresholds.setdefault("mask_nms_iou_hole", infer_cfg.get("mask_nms_iou_hole", 0.5))
    for key, value in list(thresholds.items()):
        if isinstance(value, str) and value == "auto_from_val":
            thresholds[key] = 0.5
    return {key: float(value) for key, value in thresholds.items()}


def _iter_images(single_image: str | None, image_dir: str | None) -> list[Path]:
    if single_image:
        return [Path(single_image).expanduser().resolve()]
    if image_dir:
        root = Path(image_dir).expanduser().resolve()
        return sorted([path for path in root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    raise ValueError("Either --image or --image-dir must be provided.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    infer_cfg = dict(config.get("infer", {}))

    model = WireCRHQInstSAM.from_model_config(model_cfg, image_size=int(data_cfg.get("image_size", 1024)))
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    thresholds = _load_thresholds(args.checkpoint, infer_cfg)
    inferencer = WireCRHQInstSAMInferencer(
        model=model,
        device=device,
        sliding_window=int(infer_cfg.get("sliding_window", 1024)),
        overlap=float(infer_cfg.get("overlap", 0.2)),
        score_thresh_label=thresholds["score_thresh_label"],
        score_thresh_hole=thresholds["score_thresh_hole"],
        mask_nms_iou_label=thresholds["mask_nms_iou_label"],
        mask_nms_iou_hole=thresholds["mask_nms_iou_hole"],
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    for image_path in _iter_images(args.image, args.image_dir):
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        prediction = inferencer.predict_image(image)
        inferencer.export_predictions(
            image=image,
            prediction=prediction,
            output_dir=output_dir,
            stem=image_path.stem,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
