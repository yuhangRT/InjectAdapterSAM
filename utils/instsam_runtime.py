"""
Runtime helpers shared across WireCR-InstSAM train/eval/infer scripts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.backbones.sam_backend import SAMBackendConfig, build_sam_backend
from models.wirecr_instsam import WireCRInstSAM
from utils.coco_export import CATEGORIES
from utils.instance_matcher import box_iou
from utils.instance_metrics import encode_binary_mask

__all__ = [
    "build_instsam_model",
    "resolve_instsam_run_dir",
    "save_checkpoint",
    "load_checkpoint_into_model",
    "filter_predictions",
    "predictions_to_coco",
]


def build_instsam_model(args, device: torch.device) -> WireCRInstSAM:
    backend = build_sam_backend(
        SAMBackendConfig(
            backend_type=args.sam_backend,
            sam_model_type=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint,
            fallback_sam_model_type=args.fallback_sam_model_type,
            fallback_sam_checkpoint=args.fallback_sam_checkpoint,
        )
    )
    moved = set()
    for attr_name in ("sam_model", "image_encoder", "prompt_encoder", "mask_decoder"):
        module = getattr(backend, attr_name, None)
        if isinstance(module, torch.nn.Module) and id(module) not in moved:
            module.to(device)
            moved.add(id(module))
    model = WireCRInstSAM(
        backend=backend,
        freeze_encoder=args.freeze_encoder,
        enable_roi_refiner=args.enable_roi_refiner,
        topk_per_class=args.topk_per_class,
        box_nms_iou=args.proposal_box_nms_iou,
    )
    model.to(device)
    return model


def resolve_instsam_run_dir(args) -> str:
    run_name = args.run_name or f"wirecr_instsam_{args.sam_model_type}_{args.phase if hasattr(args, 'phase') else 'eval'}"
    return os.path.join(os.path.abspath(args.save_dir), run_name)


def save_checkpoint(path: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _mask_iou(left_mask: torch.Tensor, right_mask: torch.Tensor) -> float:
    left_binary = left_mask > 0
    right_binary = right_mask > 0
    intersection = float((left_binary & right_binary).sum().item())
    union = float((left_binary | right_binary).sum().item())
    return intersection / max(union, 1.0)


def classwise_mask_nms(
    predictions: list[dict[str, Any]],
    *,
    wire_threshold: float,
    hole_threshold: float,
    wire_nms_iou: float,
    hole_nms_iou: float,
    topk_per_class: int = 50,
) -> list[dict[str, Any]]:
    by_class = {1: [], 2: []}
    for prediction in predictions:
        class_id = int(prediction["category_id"])
        threshold = wire_threshold if class_id == 1 else hole_threshold
        if float(prediction["score"]) >= threshold:
            by_class[class_id].append(prediction)

    filtered = []
    for class_id, items in by_class.items():
        items = sorted(items, key=lambda item: float(item["score"]), reverse=True)[:topk_per_class]
        keep = []
        iou_threshold = wire_nms_iou if class_id == 1 else hole_nms_iou
        for item in items:
            should_keep = True
            for kept in keep:
                if _mask_iou(item["mask_processed"], kept["mask_processed"]) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(item)
        filtered.extend(keep)
    return sorted(filtered, key=lambda item: float(item["score"]), reverse=True)


def filter_predictions(
    batched_predictions: list[list[dict[str, Any]]],
    *,
    wire_threshold: float,
    hole_threshold: float,
    wire_nms_iou: float = 0.60,
    hole_nms_iou: float = 0.60,
    topk_per_class: int = 50,
) -> list[list[dict[str, Any]]]:
    return [
        classwise_mask_nms(
            predictions,
            wire_threshold=wire_threshold,
            hole_threshold=hole_threshold,
            wire_nms_iou=wire_nms_iou,
            hole_nms_iou=hole_nms_iou,
            topk_per_class=topk_per_class,
        )
        for predictions in batched_predictions
    ]


def predictions_to_coco(
    *,
    batched_predictions: list[list[dict[str, Any]]],
    image_ids: list[int],
) -> list[dict[str, Any]]:
    coco_predictions = []
    for image_id, predictions in zip(image_ids, batched_predictions):
        for prediction in predictions:
            mask = prediction["mask_full"].detach().cpu().numpy().astype(np.uint8)
            bbox = prediction["bbox_full"].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = bbox
            coco_predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(prediction["category_id"]),
                    "score": float(prediction["score"]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "segmentation": encode_binary_mask(mask),
                }
            )
    return coco_predictions


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
