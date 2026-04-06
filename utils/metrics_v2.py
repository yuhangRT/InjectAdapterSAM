"""Evaluation metrics and threshold search helpers for WireCR-HQInstSAM."""

from __future__ import annotations

import itertools
import io
from contextlib import redirect_stdout
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from pycocotools import mask as mask_utils
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover - production environment should provide pycocotools
    mask_utils = None
    COCOeval = None

__all__ = [
    "CLASS_NAMES",
    "apply_classwise_mask_nms",
    "boundary_f1_score",
    "build_coco_predictions",
    "compute_industrial_metrics",
    "evaluate_coco_mask_metrics",
    "mask_iou",
    "search_thresholds",
]


CLASS_NAMES = {
    1: "label_sleeve",
    2: "empty_terminal",
}


def _as_bool_mask(mask: torch.Tensor | np.ndarray, *, size: tuple[int, int] | None = None) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        tensor = torch.from_numpy(mask)
    else:
        tensor = mask.detach().cpu()
    if tensor.dim() == 3:
        tensor = tensor[0]
    tensor = tensor.bool()
    if size is not None and tuple(tensor.shape[-2:]) != tuple(size):
        tensor = F.interpolate(
            tensor.unsqueeze(0).unsqueeze(0).float(),
            size=size,
            mode="nearest",
        )[0, 0] > 0.5
    return tensor


def mask_iou(mask_a: torch.Tensor | np.ndarray, mask_b: torch.Tensor | np.ndarray) -> float:
    tensor_a = _as_bool_mask(mask_a)
    tensor_b = _as_bool_mask(mask_b, size=tuple(tensor_a.shape[-2:]))
    intersection = float((tensor_a & tensor_b).float().sum().item())
    union = float((tensor_a | tensor_b).float().sum().item())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded)[0, 0] > 0.0


def boundary_f1_score(pred_mask: torch.Tensor | np.ndarray, target_mask: torch.Tensor | np.ndarray) -> float:
    pred = _as_bool_mask(pred_mask)
    target = _as_bool_mask(target_mask, size=tuple(pred.shape[-2:]))
    pred_boundary = _boundary_map(pred)
    target_boundary = _boundary_map(target)
    tp = float((pred_boundary & target_boundary).float().sum().item())
    fp = float((pred_boundary & ~target_boundary).float().sum().item())
    fn = float((~pred_boundary & target_boundary).float().sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _to_target_instances(target: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = target["labels"].detach().cpu()
    masks = target["masks"].detach().cpu()
    boxes = target.get("boxes")
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu()
    else:
        boxes = torch.zeros((labels.shape[0], 4), dtype=torch.float32)
    return [
        {
            "label": int(labels[index].item()),
            "mask": _as_bool_mask(masks[index]),
            "box": boxes[index],
        }
        for index in range(int(labels.shape[0]))
    ]


def apply_classwise_mask_nms(
    instances: Sequence[Mapping[str, Any]],
    *,
    iou_threshold_label: float,
    iou_threshold_hole: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    grouped = {
        1: sorted(
            [dict(item) for item in instances if int(item["label"]) == 1],
            key=lambda item: float(item["score"]),
            reverse=True,
        ),
        2: sorted(
            [dict(item) for item in instances if int(item["label"]) == 2],
            key=lambda item: float(item["score"]),
            reverse=True,
        ),
    }
    thresholds = {1: float(iou_threshold_label), 2: float(iou_threshold_hole)}
    for class_label, class_items in grouped.items():
        threshold = thresholds[class_label]
        while class_items:
            current = class_items.pop(0)
            kept.append(current)
            remaining = []
            for candidate in class_items:
                if mask_iou(current["mask"], candidate["mask"]) <= threshold:
                    remaining.append(candidate)
            class_items = remaining
    kept.sort(key=lambda item: float(item["score"]), reverse=True)
    return kept


def _greedy_match(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
    class_label: int | None = None,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    pred_items = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if class_label is None or int(prediction["label"]) == class_label
    ]
    gt_items = [
        (index, target)
        for index, target in enumerate(targets)
        if class_label is None or int(target["label"]) == class_label
    ]
    pred_items.sort(key=lambda item: float(item[1]["score"]), reverse=True)
    used_targets: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    unmatched_predictions: list[int] = []
    for pred_index, prediction in pred_items:
        best_gt = None
        best_iou = 0.0
        for gt_index, target in gt_items:
            if gt_index in used_targets:
                continue
            iou = mask_iou(prediction["mask"], target["mask"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_gt is not None and best_iou >= iou_threshold:
            used_targets.add(best_gt)
            matches.append((pred_index, best_gt, best_iou))
        else:
            unmatched_predictions.append(pred_index)
    unmatched_targets = [gt_index for gt_index, _ in gt_items if gt_index not in used_targets]
    return matches, unmatched_predictions, unmatched_targets


def compute_industrial_metrics(
    predictions_by_image: Sequence[Sequence[Mapping[str, Any]]],
    targets_by_image: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    mask_iou_values = []
    label_boundary_values = []
    count_errors = []
    merge_error_count = 0
    split_error_count = 0
    hole_true_positive = 0
    hole_total = 0
    class_ap50 = {1: [], 2: []}

    for predictions, target in zip(predictions_by_image, targets_by_image):
        gt_instances = _to_target_instances(target)
        count_errors.append(abs(len(predictions) - len(gt_instances)))

        for class_label in (1, 2):
            matches50, _, unmatched_targets50 = _greedy_match(
                predictions,
                gt_instances,
                iou_threshold=0.5,
                class_label=class_label,
            )
            gt_count = sum(1 for item in gt_instances if int(item["label"]) == class_label)
            pred_count = sum(1 for item in predictions if int(item["label"]) == class_label)
            tp = len(matches50)
            fp = max(pred_count - tp, 0)
            fn = max(gt_count - tp, 0)
            class_ap50[class_label].append(tp / max(tp + fp + fn, 1))
            if class_label == 2:
                hole_true_positive += tp
                hole_total += gt_count

        matches50, _, _ = _greedy_match(predictions, gt_instances, iou_threshold=0.5)
        for pred_index, gt_index, iou in matches50:
            prediction = predictions[pred_index]
            target_item = gt_instances[gt_index]
            mask_iou_values.append(iou)
            if int(target_item["label"]) == 1:
                label_boundary_values.append(boundary_f1_score(prediction["mask"], target_item["mask"]))

        for prediction in predictions:
            same_class_targets = [item for item in gt_instances if int(item["label"]) == int(prediction["label"])]
            overlaps = sum(mask_iou(prediction["mask"], target_item["mask"]) >= 0.1 for target_item in same_class_targets)
            if overlaps > 1:
                merge_error_count += 1
        for target_item in gt_instances:
            same_class_predictions = [item for item in predictions if int(item["label"]) == int(target_item["label"])]
            overlaps = sum(mask_iou(prediction["mask"], target_item["mask"]) >= 0.1 for prediction in same_class_predictions)
            if overlaps > 1:
                split_error_count += 1

    return {
        "empty_terminal_recall": float(hole_true_positive / max(hole_total, 1)),
        "label_sleeve_boundary_f1": float(np.mean(label_boundary_values) if label_boundary_values else 0.0),
        "mean_mask_iou": float(np.mean(mask_iou_values) if mask_iou_values else 0.0),
        "count_mae": float(np.mean(count_errors) if count_errors else 0.0),
        "merge_error_count": float(merge_error_count),
        "split_error_count": float(split_error_count),
        "label_sleeve_ap50": float(np.mean(class_ap50[1]) if class_ap50[1] else 0.0),
        "empty_terminal_ap50": float(np.mean(class_ap50[2]) if class_ap50[2] else 0.0),
    }


def encode_binary_mask(mask: torch.Tensor | np.ndarray) -> dict[str, Any]:
    if mask_utils is None:
        raise ImportError("pycocotools is required to encode binary masks")
    bool_mask = _as_bool_mask(mask).numpy().astype(np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(bool_mask))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def build_coco_predictions(
    predictions_by_image: Sequence[Sequence[Mapping[str, Any]]],
    image_ids: Sequence[int],
) -> list[dict[str, Any]]:
    coco_predictions = []
    for image_id, predictions in zip(image_ids, predictions_by_image):
        for prediction in predictions:
            coco_predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(prediction["label"]),
                    "segmentation": encode_binary_mask(prediction["mask"]),
                    "score": float(prediction["score"]),
                }
            )
    return coco_predictions


def evaluate_coco_mask_metrics(coco_gt: Any, predictions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if coco_gt is None or COCOeval is None:
        return {
            "mask_ap": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "per_class_AP50": {
                "label_sleeve": 0.0,
                "empty_terminal": 0.0,
            },
        }

    if not predictions:
        return {
            "mask_ap": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "per_class_AP50": {
                "label_sleeve": 0.0,
                "empty_terminal": 0.0,
            },
        }

    coco_dt = coco_gt.loadRes(list(predictions))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    with redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    per_class_ap50 = {}
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    cat_ids = list(coco_gt.getCatIds())
    for class_index, cat_id in enumerate(cat_ids):
        class_name = CLASS_NAMES.get(int(cat_id), str(cat_id))
        class_precision = precision[0, :, class_index, 0, -1]
        valid = class_precision[class_precision > -1]
        per_class_ap50[class_name] = float(valid.mean()) if valid.size else 0.0

    return {
        "mask_ap": float(coco_eval.stats[0]),
        "mask_ap50": float(coco_eval.stats[1]),
        "mask_ap75": float(coco_eval.stats[2]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "per_class_AP50": per_class_ap50,
    }


def search_thresholds(
    records: Sequence[Mapping[str, Any]],
    *,
    coco_gt: Any = None,
    selector,
    coco_selector=None,
    score_grid_label: Sequence[float],
    score_grid_hole: Sequence[float],
    nms_grid_label: Sequence[float],
    nms_grid_hole: Sequence[float],
) -> dict[str, Any]:
    search_results = []
    best_summary = None

    for score_thresh_label, score_thresh_hole, mask_nms_iou_label, mask_nms_iou_hole in itertools.product(
        score_grid_label,
        score_grid_hole,
        nms_grid_label,
        nms_grid_hole,
    ):
        thresholds = {
            "score_thresh_label": float(score_thresh_label),
            "score_thresh_hole": float(score_thresh_hole),
            "mask_nms_iou_label": float(mask_nms_iou_label),
            "mask_nms_iou_hole": float(mask_nms_iou_hole),
        }
        predictions_by_image = [selector(record, thresholds) for record in records]
        coco_predictions_by_image = (
            [coco_selector(record, thresholds) for record in records]
            if coco_selector is not None
            else predictions_by_image
        )
        targets_by_image = [record["target"] for record in records]
        image_ids = [int(record["image_id"]) for record in records]
        if coco_gt is None:
            coco_metrics = evaluate_coco_mask_metrics(None, [])
        else:
            coco_metrics = evaluate_coco_mask_metrics(coco_gt, build_coco_predictions(coco_predictions_by_image, image_ids))
        industrial_metrics = compute_industrial_metrics(predictions_by_image, targets_by_image)
        summary = {
            **thresholds,
            **coco_metrics,
            **industrial_metrics,
        }
        search_results.append(summary)

        ranking_key = (
            float(summary["mask_ap"]),
            float(summary["AP50"]),
            float(summary["empty_terminal_recall"]),
            -float(summary["count_mae"]),
        )
        if best_summary is None or ranking_key > best_summary[0]:
            best_summary = (ranking_key, summary)

    assert best_summary is not None
    return {
        "best_thresholds": {
            "score_thresh_label": best_summary[1]["score_thresh_label"],
            "score_thresh_hole": best_summary[1]["score_thresh_hole"],
            "mask_nms_iou_label": best_summary[1]["mask_nms_iou_label"],
            "mask_nms_iou_hole": best_summary[1]["mask_nms_iou_hole"],
        },
        "best_metrics": best_summary[1],
        "search_results": search_results,
    }
