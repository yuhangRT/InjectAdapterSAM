"""Tests for the S06 prompt builder."""

from __future__ import annotations

import math

import torch

from models.prompt_builder_v2 import PromptBuilderV2
from models.wirecr_hq_instsam import WireCRHQInstSAM


def _build_prompt_builder() -> PromptBuilderV2:
    return PromptBuilderV2(
        dense_prompt_downscale=4,
        gt_box_jitter=0.05,
        joint_gt_ratio_start=0.7,
        joint_gt_ratio_end=0.1,
    )


def _build_test_model(*, num_queries: int = 4) -> WireCRHQInstSAM:
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
        num_queries=num_queries,
        query_decoder_layers=2,
    )


def _pred_instances() -> dict[str, torch.Tensor | tuple[int, int]]:
    mask_logits = torch.linspace(-2.5, 2.5, steps=2 * 16 * 16, dtype=torch.float32).reshape(2, 16, 16)
    return {
        "boxes": torch.tensor(
            [
                [8.0, 12.0, 48.0, 28.0],
                [20.0, 20.0, 44.0, 44.0],
            ],
            dtype=torch.float32,
        ),
        "mask_logits": mask_logits,
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "oriented_boxes": torch.tensor(
            [
                [28.0, 20.0, 40.0, 16.0, 0.0],
                [32.0, 32.0, 24.0, 24.0, 45.0],
            ],
            dtype=torch.float32,
        ),
        "principal_axes": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "processed_size": (64, 64),
    }


def _gt_instances() -> dict[str, torch.Tensor | tuple[int, int]]:
    masks = torch.zeros((2, 64, 64), dtype=torch.float32)
    masks[0, 12:28, 8:48] = 1.0
    masks[1, 20:44, 20:44] = 1.0
    return {
        "boxes": torch.tensor(
            [
                [8.0, 12.0, 48.0, 28.0],
                [20.0, 20.0, 44.0, 44.0],
            ],
            dtype=torch.float32,
        ),
        "masks": masks,
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "oriented_boxes": torch.tensor(
            [
                [28.0, 20.0, 40.0, 16.0, 0.0],
                [32.0, 32.0, 24.0, 24.0, 30.0],
            ],
            dtype=torch.float32,
        ),
        "principal_axes": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "processed_size": (64, 64),
    }


def test_prompt_builder_pred_path_uses_class_specific_rules() -> None:
    builder = _build_prompt_builder()
    prompts = builder.build_prompts(
        pred_instances=_pred_instances(),
        processed_size=(64, 64),
        prompt_source="pred",
    )

    assert prompts["boxes_xyxy"].shape == (2, 4)
    assert prompts["dense_mask_prompt_logits"].shape == (2, 1, 16, 16)
    assert prompts["point_coords"].shape == (2, 5, 2)
    assert prompts["point_labels"].shape == (2, 5)
    assert torch.allclose(prompts["boxes_xyxy"], _pred_instances()["boxes"])
    assert not torch.all((prompts["dense_mask_prompt_logits"] == 0) | (prompts["dense_mask_prompt_logits"] == 1))

    label_meta = prompts["prompt_meta"][0]
    hole_meta = prompts["prompt_meta"][1]
    assert label_meta["prompt_source"] == "pred"
    assert label_meta["sampling_strategy"] == "label_axis_3pos_2neg"
    assert label_meta["used_principal_axis_aux"] is True
    assert prompts["point_labels"][0].tolist() == [1, 1, 1, 0, 0]

    assert hole_meta["sampling_strategy"] == "hole_center_1pos_4neg"
    assert prompts["point_labels"][1].tolist() == [1, 0, 0, 0, 0]


def test_prompt_builder_pred_path_records_gt_matches_when_available() -> None:
    builder = _build_prompt_builder()
    prompts = builder.build_prompts(
        pred_instances=_pred_instances(),
        gt_instances=_gt_instances(),
        processed_size=(64, 64),
        prompt_source="pred",
    )

    assert [meta["matched_gt_index"] for meta in prompts["prompt_meta"]] == [0, 1]
    assert all(meta["matched_gt_box_iou"] > 0.99 for meta in prompts["prompt_meta"])


def test_prompt_builder_gt_path_generates_low_res_logits_and_jitter() -> None:
    builder = _build_prompt_builder()
    generator = torch.Generator().manual_seed(7)
    gt_instances = _gt_instances()

    prompts = builder.build_prompts(
        gt_instances=gt_instances,
        processed_size=(64, 64),
        prompt_source="gt",
        generator=generator,
    )

    assert prompts["dense_mask_prompt_logits"].shape == (2, 1, 16, 16)
    assert not torch.allclose(prompts["boxes_xyxy"], gt_instances["boxes"])
    assert all(meta["box_jitter_applied"] for meta in prompts["prompt_meta"])
    assert all(meta["instance_source"] == "gt" for meta in prompts["prompt_meta"])
    dense_prompt = prompts["dense_mask_prompt_logits"][0]
    assert dense_prompt.shape[-2:] != gt_instances["masks"].shape[-2:]
    assert torch.isfinite(dense_prompt).all()
    assert not torch.all((dense_prompt == 0) | (dense_prompt == 1))


def test_prompt_builder_mixed_path_tracks_sources_and_ratio_schedule() -> None:
    builder = _build_prompt_builder()
    assert math.isclose(builder.resolve_joint_gt_ratio(0.0), 0.7, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(builder.resolve_joint_gt_ratio(1.0), 0.1, rel_tol=0.0, abs_tol=1e-6)

    prompts = builder.build_prompts(
        pred_instances=_pred_instances(),
        gt_instances=_gt_instances(),
        processed_size=(64, 64),
        prompt_source="mixed",
        gt_ratio=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert all(meta["prompt_source"] == "mixed" for meta in prompts["prompt_meta"])
    assert all(meta["instance_source"] == "gt" for meta in prompts["prompt_meta"])


def test_top_level_model_forward_prompt_builder_runs_for_gt_pred_and_mixed() -> None:
    model = _build_test_model(num_queries=4)
    with torch.no_grad():
        model.query_head.class_embed.weight.zero_()
        model.query_head.class_embed.bias.zero_()
        model.query_head.class_embed.bias[1] = 6.0

    image = torch.rand(1, 3, 32, 40)
    targets = [_gt_instances()]

    pred_outputs = model.forward_prompt_builder(image, prompt_source="pred")
    gt_outputs = model.forward_prompt_builder(
        image,
        targets=targets,
        prompt_source="gt",
        generator=torch.Generator().manual_seed(1),
    )
    mixed_outputs = model.forward_prompt_builder(
        image,
        targets=targets,
        prompt_source="mixed",
        gt_ratio=1.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert len(pred_outputs["prompt_batches"]) == 1
    assert len(gt_outputs["prompt_batches"]) == 1
    assert len(mixed_outputs["prompt_batches"]) == 1
    assert pred_outputs["prompt_batches"][0]["prompt_meta"]
    assert gt_outputs["prompt_batches"][0]["prompt_meta"][0]["prompt_source"] == "gt"
    assert mixed_outputs["prompt_batches"][0]["prompt_meta"][0]["prompt_source"] == "mixed"
    assert mixed_outputs["prompt_batches"][0]["prompt_meta"][0]["instance_source"] == "gt"
