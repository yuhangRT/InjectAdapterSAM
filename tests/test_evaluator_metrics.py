"""Evaluator and metrics smoke tests for S10."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO

from engine.evaluator import WireCRHQInstSAMEvaluator
from utils.metrics_v2 import encode_binary_mask, evaluate_coco_mask_metrics


class _FakeEvalModel(nn.Module):
    def forward(self, images: torch.Tensor, *, processed_sizes, prompt_source="pred", **kwargs):
        fused_batches = []
        for batch_index in range(int(images.shape[0])):
            image = images[batch_index]
            mask = image[0] > 0.5
            ys, xs = torch.where(mask)
            if xs.numel() == 0:
                fused_batches.append(
                    {
                        "labels": torch.zeros((0,), dtype=torch.long),
                        "instance_scores": torch.zeros((0,), dtype=torch.float32),
                        "refined_mask_logits": torch.zeros((0, 1, 16, 16), dtype=torch.float32),
                        "boxes_xyxy": torch.zeros((0, 4), dtype=torch.float32),
                    }
                )
                continue
            box = torch.tensor([[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]], dtype=torch.float32)
            logits = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(16, 16), mode="nearest") * 12.0 - 6.0
            fused_batches.append(
                {
                    "labels": torch.tensor([1], dtype=torch.long),
                    "instance_scores": torch.tensor([0.95], dtype=torch.float32),
                    "refined_mask_logits": logits,
                    "boxes_xyxy": box,
                }
            )
        return {
            "eval_dict": {
                "fused_batches": fused_batches,
            }
        }


def _sample_batch(seed: int) -> dict[str, object]:
    image = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    image[:, :, 16:48, 16:48] = 1.0
    mask = torch.zeros((1, 64, 64), dtype=torch.uint8)
    mask[:, 16:48, 16:48] = 1
    return {
        "image": image,
        "image_id": [seed],
        "image_path": [f"sample_{seed}.png"],
        "orig_size": [(64, 64)],
        "processed_size": [(64, 64)],
        "crop_box": [None],
        "crop_mode": ["full"],
        "instances": [
            {
                "boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
                "masks": mask,
                "areas": torch.tensor([float(mask.sum())], dtype=torch.float32),
                "iscrowd": torch.zeros((1,), dtype=torch.long),
                "group_ids": torch.zeros((1,), dtype=torch.long),
                "oriented_boxes": torch.tensor([[32.0, 32.0, 32.0, 32.0, 0.0]], dtype=torch.float32),
                "principal_axes": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            }
        ],
    }


def test_evaluator_threshold_search_and_metrics_json(tmp_path: Path) -> None:
    loader = [_sample_batch(1), _sample_batch(2)]
    evaluator = WireCRHQInstSAMEvaluator(
        model=_FakeEvalModel(),
        device="cpu",
        score_grid_label=(0.2, 0.5),
        score_grid_hole=(0.2,),
        nms_grid_label=(0.5,),
        nms_grid_hole=(0.5,),
    )

    result = evaluator.evaluate(loader, output_path=tmp_path / "metrics.json")
    assert "best_thresholds" in result
    assert "best_metrics" in result
    assert "search_results" in result
    assert result["best_metrics"]["label_sleeve_ap50"] >= 0.0
    assert result["best_metrics"]["merge_error_count"] >= 0.0
    assert (tmp_path / "metrics.json").is_file()


def test_evaluator_projects_processed_masks_back_to_original_space_for_coco_ap(tmp_path: Path) -> None:
    orig_height, orig_width = 40, 80
    processed_size = (64, 64)
    mask = torch.zeros((orig_height, orig_width), dtype=torch.uint8)
    mask[10:30, 20:60] = 1

    scale = min(processed_size[1] / float(orig_width), processed_size[0] / float(orig_height))
    resized_width = int(round(orig_width * scale))
    resized_height = int(round(orig_height * scale))
    pad_left = (processed_size[1] - resized_width) // 2
    pad_top = (processed_size[0] - resized_height) // 2
    processed_mask = F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=(resized_height, resized_width),
        mode="nearest",
    )[0, 0]
    padded_mask = torch.zeros(processed_size, dtype=torch.float32)
    padded_mask[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = processed_mask
    logits = padded_mask.unsqueeze(0).unsqueeze(0) * 12.0 - 6.0

    coco_dict = {
        "images": [{"id": 1, "file_name": "sample.png", "height": orig_height, "width": orig_width}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[20, 10, 60, 10, 60, 30, 20, 30]],
                "area": 800.0,
                "bbox": [20.0, 10.0, 40.0, 20.0],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 1, "name": "label_sleeve"},
            {"id": 2, "name": "empty_terminal"},
        ],
    }
    ann_path = tmp_path / "instances_val.json"
    ann_path.write_text(json.dumps(coco_dict), encoding="utf-8")
    coco = COCO(str(ann_path))

    evaluator = WireCRHQInstSAMEvaluator(
        model=_FakeEvalModel(),
        device="cpu",
        score_grid_label=(0.2,),
        score_grid_hole=(0.2,),
        nms_grid_label=(0.5,),
        nms_grid_hole=(0.5,),
    )

    record = {
        "image_id": 1,
        "image_path": "sample.png",
        "orig_size": (orig_height, orig_width),
        "processed_size": processed_size,
        "crop_box": None,
        "fused_batch": {
            "labels": torch.tensor([1], dtype=torch.long),
            "instance_scores": torch.tensor([0.95], dtype=torch.float32),
            "refined_mask_logits": logits,
            "boxes_xyxy": torch.tensor([[20.0, 16.0, 44.0, 48.0]], dtype=torch.float32),
        },
        "target": {
            "boxes": torch.tensor([[20.0, 16.0, 44.0, 48.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "masks": padded_mask.unsqueeze(0).to(torch.uint8),
        },
    }
    thresholds = {
        "score_thresh_label": 0.2,
        "score_thresh_hole": 0.2,
        "mask_nms_iou_label": 0.5,
        "mask_nms_iou_hole": 0.5,
    }

    coco_instances = evaluator._select_coco_instances_from_record(record, thresholds)
    assert len(coco_instances) == 1
    predictions = [
        {
            "image_id": 1,
            "category_id": int(coco_instances[0]["label"]),
            "segmentation": encode_binary_mask(coco_instances[0]["mask"]),
            "score": float(coco_instances[0]["score"]),
        }
    ]

    coco_metrics = evaluate_coco_mask_metrics(coco, predictions)
    assert coco_metrics["mask_ap50"] >= 0.99
