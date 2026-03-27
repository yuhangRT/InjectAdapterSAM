"""
Evaluation metrics for instance segmentation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from utils.instance_matcher import box_iou, greedy_match_boxes
from utils.sam_metrics import compute_boundary_f1, compute_cldice

try:
    from pycocotools import mask as mask_utils
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover
    mask_utils = None
    COCOeval = None

__all__ = [
    "encode_binary_mask",
    "coco_eval_from_predictions",
    "compute_industrial_instance_metrics",
]


def _prediction_mask_tensor(prediction: dict[str, Any]) -> torch.Tensor:
    for key in ("mask", "mask_processed", "mask_full"):
        if key in prediction:
            return prediction[key]
    raise KeyError("Prediction is missing a mask-like field. Expected one of: mask, mask_processed, mask_full.")


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    if mask_utils is None:
        raise ImportError("pycocotools is required to encode masks.")
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def coco_eval_from_predictions(coco_gt, predictions: list[dict[str, Any]]) -> dict[str, float]:
    if COCOeval is None:
        raise ImportError("pycocotools is required to evaluate COCO metrics.")
    if not predictions:
        return {key: 0.0 for key in ("AP", "AP50", "AP75", "APS", "APM", "APL")}
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ("AP", "AP50", "AP75", "APS", "APM", "APL")
    return {key: float(value) for key, value in zip(keys, evaluator.stats[:6])}


def _pair_predictions(
    predictions: list[dict[str, Any]],
    target_instances: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> list[tuple[dict[str, Any], int]]:
    matches = greedy_match_boxes(predictions, target_instances, iou_threshold=iou_threshold)
    return [(predictions[pred_idx], gt_idx) for pred_idx, gt_idx in matches]


def compute_industrial_instance_metrics(
    *,
    batched_predictions: list[list[dict[str, Any]]],
    batched_targets: list[dict[str, Any]],
) -> dict[str, float]:
    wire_cldice_scores = []
    boundary_f1_scores = []
    hole_recall_hits = 0
    hole_recall_total = 0
    count_errors = []
    merge_errors = []
    split_errors = []

    for predictions, targets in zip(batched_predictions, batched_targets):
        pairs = _pair_predictions(predictions, targets, iou_threshold=0.5)
        predicted_by_class = {1: 0, 2: 0}
        gt_by_class = {1: 0, 2: 0}
        for prediction in predictions:
            predicted_by_class[int(prediction["category_id"])] += 1
        for label in targets["labels"].tolist():
            gt_by_class[int(label)] += 1
        count_errors.append(abs(sum(predicted_by_class.values()) - sum(gt_by_class.values())))
        merge_errors.append(max(predicted_by_class[1] + predicted_by_class[2] - len(pairs), 0))
        split_errors.append(max(sum(gt_by_class.values()) - len(pairs), 0))

        for prediction, gt_idx in pairs:
            pred_mask = _prediction_mask_tensor(prediction).float().unsqueeze(0)
            gt_mask = targets["masks"][gt_idx].float().unsqueeze(0)
            boundary_f1_scores.append(float(compute_boundary_f1(pred_mask, gt_mask).mean().item()))
            label = int(targets["labels"][gt_idx].item())
            if label == 1:
                wire_cldice_scores.append(float(compute_cldice(pred_mask, gt_mask).mean().item()))
            else:
                hole_recall_hits += 1
        hole_recall_total += int((targets["labels"] == 2).sum().item())

    return {
        "wire_cldice": float(np.mean(wire_cldice_scores)) if wire_cldice_scores else 0.0,
        "instance_boundary_f1": float(np.mean(boundary_f1_scores)) if boundary_f1_scores else 0.0,
        "hole_recall": hole_recall_hits / max(hole_recall_total, 1),
        "count_mae": float(np.mean(count_errors)) if count_errors else 0.0,
        "merge_error_rate": float(np.mean(merge_errors)) if merge_errors else 0.0,
        "split_error_rate": float(np.mean(split_errors)) if split_errors else 0.0,
    }
