"""
Shared CLI helpers for WireCR-InstSAM scripts.
"""

from __future__ import annotations

import argparse


def add_common_instsam_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=str, required=True, help="COCO-style instance dataset root.")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="Checkpoint path for the active SAM backend.")
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="hiera_small",
        help="Requested SAM backbone type. Use hiera_small by default; SAM1 fallback uses --fallback-sam-model-type.",
    )
    parser.add_argument(
        "--sam-backend",
        type=str,
        default="auto",
        choices=["auto", "sam1", "sam2.1", "sam2", "hiera"],
        help="Preferred SAM backend. auto tries SAM2.1 first and falls back to SAM1.",
    )
    parser.add_argument(
        "--fallback-sam-model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="Fallback SAM1 model type when SAM2.1 is unavailable.",
    )
    parser.add_argument(
        "--fallback-sam-checkpoint",
        type=str,
        default=None,
        help="Optional SAM1 checkpoint path used when SAM2.1 fallback is triggered.",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true", default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--channels-last", action="store_true", default=False)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--enable-roi-refiner", action="store_true", default=False)
    parser.add_argument("--topk-per-class", type=int, default=64)
    parser.add_argument("--proposal-box-nms-iou", type=float, default=0.5)
    parser.add_argument("--hole-positive-weight", type=float, default=5.0)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--run-name", type=str, default=None)
