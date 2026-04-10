"""Standalone evaluation entrypoint for WireCR-HQInstSAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset.label_hole_instance_dataset import build_label_hole_dataloader
from engine.evaluator import WireCRHQInstSAMEvaluator
from models.wirecr_hq_instsam import WireCRHQInstSAM
from utils.checkpoint import load_checkpoint
from utils.config import load_yaml_config, merge_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WireCR-HQInstSAM on a validation split.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    parser.add_argument("--data-root", default=None, help="Optional data root override.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    if args.data_root is not None:
        config = merge_config(config, {"data.root": args.data_root})

    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    runtime_cfg = dict(config.get("runtime", {}))
    eval_cfg = dict(config.get("eval", {}))
    dataset_passthrough_keys = (
        "full_image_prob",
        "object_crop_prob",
        "hole_focused_prob",
        "label_focused_prob",
        "hole_scale_range",
        "label_scale_range",
        "object_scale_range",
        "min_crop_size",
    )
    dataset_kwargs = {
        key: data_cfg[key]
        for key in dataset_passthrough_keys
        if key in data_cfg
    }
    eval_dataset_kwargs = {
        **dataset_kwargs,
        "image_size": int(data_cfg.get("image_size", 1024)),
        "augment": False,
        "seed": int(runtime_cfg.get("seed", 42)),
        "full_image_prob": 1.0,
        "object_crop_prob": 0.0,
        "hole_focused_prob": 0.0,
        "label_focused_prob": 0.0,
    }
    gpu_index = int(runtime_cfg.get("gpu", 0))
    device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"

    dataloader = build_label_hole_dataloader(
        data_root=data_cfg["root"],
        split=args.split,
        batch_size=int(config.get("train", {}).get("val_batch_size", config.get("train", {}).get("batch_size", 1))),
        num_workers=int(config.get("train", {}).get("workers", 0)),
        **eval_dataset_kwargs,
    )
    model = WireCRHQInstSAM.from_model_config(model_cfg, image_size=int(data_cfg.get("image_size", 1024)))
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.to(device)

    evaluator = WireCRHQInstSAMEvaluator(
        model=model,
        device=device,
        score_grid_label=tuple(eval_cfg.get("score_grid_label", (0.2, 0.35, 0.5, 0.65))),
        score_grid_hole=tuple(eval_cfg.get("score_grid_hole", (0.2, 0.35, 0.5, 0.65))),
        mask_prob_grid=tuple(eval_cfg.get("mask_prob_grid", (0.4, 0.5, 0.6))),
        nms_grid_label=tuple(eval_cfg.get("mask_nms_iou_label_grid", (0.5, 0.6, 0.7))),
        nms_grid_hole=tuple(eval_cfg.get("mask_nms_iou_hole_grid", (0.45, 0.55, 0.65))),
    )
    result = evaluator.evaluate(dataloader, output_path=args.output)

    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
