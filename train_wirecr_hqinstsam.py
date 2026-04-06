"""Unified training entrypoint for WireCR-HQInstSAM."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from dataset.label_hole_instance_dataset import build_label_hole_dataloader
from engine.evaluator import WireCRHQInstSAMEvaluator
from engine.trainer import WireCRHQInstSAMTrainer
from models.wirecr_hq_instsam import WireCRHQInstSAM
from utils.config import build_config_parser, load_config_from_args, save_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(runtime_cfg: dict[str, object]) -> torch.device:
    gpu_index = runtime_cfg.get("gpu", 0)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_index)}")
    return torch.device("cpu")


def build_trainer_from_config(config: dict[str, object]) -> WireCRHQInstSAMTrainer:
    runtime_cfg = dict(config.get("runtime", {}))
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    train_cfg = dict(config.get("train", {}))
    infer_cfg = dict(config.get("infer", {}))
    eval_cfg = dict(config.get("eval", {}))

    seed = int(runtime_cfg.get("seed", 42))
    _set_seed(seed)
    device = _resolve_device(runtime_cfg)

    output_dir = Path(runtime_cfg.get("output_dir", "./runs/wirecr_hqinstsam")).expanduser().resolve()
    save_config(config, output_dir)

    data_root = data_cfg["root"]
    image_size = int(data_cfg.get("image_size", 1024))
    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "val"))
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
    base_dataset_kwargs = {
        key: data_cfg[key]
        for key in dataset_passthrough_keys
        if key in data_cfg
    }
    train_workers = int(train_cfg.get("workers", 0))
    val_workers = int(train_cfg.get("val_workers", train_workers))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", train_workers > 0))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2)) if max(train_workers, val_workers) > 0 else None

    train_dataset_kwargs = {
        **base_dataset_kwargs,
        "image_size": image_size,
        "seed": seed,
    }
    train_loader = build_label_hole_dataloader(
        data_root=data_root,
        split=train_split,
        batch_size=int(train_cfg.get("batch_size", 2)),
        num_workers=train_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        **train_dataset_kwargs,
    )
    val_dataset_kwargs = {
        **base_dataset_kwargs,
        "image_size": image_size,
        "seed": seed + 1,
        "augment": False,
        "full_image_prob": 1.0,
        "object_crop_prob": 0.0,
        "hole_focused_prob": 0.0,
        "label_focused_prob": 0.0,
    }
    val_loader = build_label_hole_dataloader(
        data_root=data_root,
        split=val_split,
        batch_size=int(train_cfg.get("val_batch_size", train_cfg.get("batch_size", 2))),
        num_workers=val_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        **val_dataset_kwargs,
    )

    model = WireCRHQInstSAM.from_model_config(model_cfg, image_size=image_size)
    evaluator = WireCRHQInstSAMEvaluator(
        model=model,
        device=device,
        score_grid_label=tuple(eval_cfg.get("score_grid_label", (0.2, 0.35, 0.5, 0.65))),
        score_grid_hole=tuple(eval_cfg.get("score_grid_hole", (0.2, 0.35, 0.5, 0.65))),
        nms_grid_label=tuple(eval_cfg.get("mask_nms_iou_label_grid", (0.5, 0.6, 0.7))),
        nms_grid_hole=tuple(eval_cfg.get("mask_nms_iou_hole_grid", (0.45, 0.55, 0.65))),
    )
    trainer = WireCRHQInstSAMTrainer(
        model=model,
        output_dir=output_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(train_cfg.get("epochs", 60)),
        warmup_epochs=int(train_cfg.get("warmup_epochs", 8)),
        lr_lora=float(train_cfg.get("lr_lora", 5e-5)),
        lr_new_modules=float(train_cfg.get("lr_new_modules", train_cfg.get("lr", 2e-4))),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        warmup_iters=int(train_cfg.get("warmup_iters", 1000)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 2)),
        max_grad_norm=float(train_cfg.get("grad_clip_norm", 0.1)),
        val_interval=int(train_cfg.get("val_interval", 1)),
        amp=bool(train_cfg.get("amp", True)),
        evaluator=evaluator,
        config_snapshot=config,
        infer_config=infer_cfg,
        resume_path=runtime_cfg.get("resume"),
    )
    return trainer


def main(argv: list[str] | None = None) -> int:
    parser = build_config_parser(description="Train WireCR-HQInstSAM with the unified trainer.")
    args = parser.parse_args(argv)
    config = load_config_from_args(args)
    trainer = build_trainer_from_config(config)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
