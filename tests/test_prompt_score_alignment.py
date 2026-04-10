"""Prompt-score alignment regression tests for S13 P0."""

from __future__ import annotations

import torch

from models.wirecr_hq_instsam import WireCRHQInstSAM


def _build_test_model() -> WireCRHQInstSAM:
    return WireCRHQInstSAM(
        model_type="vit_b",
        image_size=64,
        prompt_embed_dim=32,
        feature_dim=32,
        encoder_embed_dim=64,
        encoder_depth=12,
        encoder_num_heads=4,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        wirecr_out_channels=256,
        pixel_decoder_channels=256,
        num_queries=4,
        query_decoder_layers=2,
        checkpoint=None,
    )


def test_gather_prompt_aligned_scores_aligns_pred_sources_and_marks_gt_missing() -> None:
    model = _build_test_model()
    coarse_batch = {
        "class_scores": torch.tensor([0.11, 0.22, 0.33], dtype=torch.float32),
        "box_quality": torch.tensor([0.41, 0.52, 0.63], dtype=torch.float32),
        "coarse_mask_score": torch.tensor([0.71, 0.82, 0.93], dtype=torch.float32),
    }
    prompt_meta = [
        {"instance_source": "pred", "source_index": 2},
        {"instance_source": "pred", "source_index": 0},
        {"instance_source": "pred", "source_index": 1},
        {"instance_source": "gt", "source_index": 0},
    ]

    class_scores, box_quality, coarse_mask_score, coarse_score_missing = model._gather_prompt_aligned_scores(
        prompt_meta=prompt_meta,
        coarse_batch=coarse_batch,
        device=torch.device("cpu"),
    )

    assert torch.allclose(class_scores, torch.tensor([0.33, 0.11, 0.22, 0.0], dtype=torch.float32))
    assert torch.allclose(box_quality, torch.tensor([0.63, 0.41, 0.52, 0.0], dtype=torch.float32))
    assert torch.allclose(coarse_mask_score, torch.tensor([0.93, 0.71, 0.82, 0.0], dtype=torch.float32))
    assert torch.equal(coarse_score_missing, torch.tensor([False, False, False, True]))
