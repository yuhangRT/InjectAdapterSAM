"""Mask threshold path regression tests for model, evaluator, and inferencer."""

from __future__ import annotations

import torch

from engine.evaluator import WireCRHQInstSAMEvaluator
from engine.inferencer import WireCRHQInstSAMInferencer


def _threshold_record(logit_value: float) -> dict[str, object]:
    return {
        "processed_size": (16, 16),
        "orig_size": (16, 16),
        "crop_box": None,
        "fused_batch": {
            "labels": torch.tensor([1], dtype=torch.long),
            "instance_scores": torch.tensor([0.95], dtype=torch.float32),
            "refined_mask_logits": torch.full((1, 1, 16, 16), logit_value, dtype=torch.float32),
            "boxes_xyxy": torch.tensor([[2.0, 2.0, 14.0, 14.0]], dtype=torch.float32),
        },
        "target": {
            "labels": torch.tensor([1], dtype=torch.long),
            "masks": torch.ones((1, 16, 16), dtype=torch.uint8),
            "boxes": torch.tensor([[0.0, 0.0, 16.0, 16.0]], dtype=torch.float32),
        },
        "image_id": 1,
    }


def test_mask_prob_thresh_half_matches_logit_zero_boundary() -> None:
    logits = torch.tensor([-1.5, -0.2, 0.0, 0.2, 1.5], dtype=torch.float32)
    assert torch.equal(logits.sigmoid() > 0.5, logits > 0.0)


def test_evaluator_selector_respects_mask_prob_threshold() -> None:
    record = _threshold_record(0.2)
    thresholds_lo = {
        "score_thresh_label": 0.1,
        "score_thresh_hole": 0.1,
        "mask_prob_thresh": 0.5,
        "mask_nms_iou_label": 0.6,
        "mask_nms_iou_hole": 0.5,
    }
    thresholds_hi = dict(thresholds_lo)
    thresholds_hi["mask_prob_thresh"] = 0.7

    selected_lo = WireCRHQInstSAMEvaluator._select_instances_from_record(
        record,
        thresholds_lo,
        output_space="processed",
    )
    selected_hi = WireCRHQInstSAMEvaluator._select_instances_from_record(
        record,
        thresholds_hi,
        output_space="processed",
    )

    assert len(selected_lo) == 1
    assert selected_hi == []


def test_inferencer_window_instances_respect_mask_prob_threshold() -> None:
    inferencer_lo = WireCRHQInstSAMInferencer(
        model=torch.nn.Identity(),
        device="cpu",
        score_thresh_label=0.1,
        score_thresh_hole=0.1,
        mask_prob_thresh=0.5,
    )
    inferencer_hi = WireCRHQInstSAMInferencer(
        model=torch.nn.Identity(),
        device="cpu",
        score_thresh_label=0.1,
        score_thresh_hole=0.1,
        mask_prob_thresh=0.7,
    )
    fused_batch = {
        "labels": torch.tensor([1], dtype=torch.long),
        "instance_scores": torch.tensor([0.95], dtype=torch.float32),
        "refined_mask_logits": torch.full((1, 1, 16, 16), 0.2, dtype=torch.float32),
        "boxes_xyxy": torch.tensor([[2.0, 2.0, 14.0, 14.0]], dtype=torch.float32),
    }

    selected_lo = inferencer_lo._fused_batch_to_window_instances(
        fused_batch,
        crop_size=(16, 16),
        window_box=(0, 0, 16, 16),
        full_size=(16, 16),
    )
    selected_hi = inferencer_hi._fused_batch_to_window_instances(
        fused_batch,
        crop_size=(16, 16),
        window_box=(0, 0, 16, 16),
        full_size=(16, 16),
    )

    assert len(selected_lo) == 1
    assert selected_hi == []
