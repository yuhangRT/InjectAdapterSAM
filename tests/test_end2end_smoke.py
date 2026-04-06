"""End-to-end smoke tests for S08 closed-loop inference."""

from __future__ import annotations

import torch

from models.mask_nms import class_wise_mask_nms
from models.score_fusion import ScoreFusion
from models.wirecr_hq_instsam import WireCRHQInstSAM


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


def test_score_fusion_and_classwise_mask_nms_behave_as_explicit_modules() -> None:
    fusion = ScoreFusion(hidden_dim=8)
    fused = fusion(
        class_score=torch.tensor([0.8, 0.6], dtype=torch.float32),
        box_quality=torch.tensor([0.9, 0.7], dtype=torch.float32),
        coarse_mask_score=torch.tensor([0.7, 0.5], dtype=torch.float32),
        refine_quality_score=torch.tensor([0.85, 0.4], dtype=torch.float32),
    )
    naive = torch.tensor([0.8 * 0.9 * 0.7 * 0.85, 0.6 * 0.7 * 0.5 * 0.4], dtype=torch.float32)
    assert torch.isfinite(fused).all()
    assert not torch.allclose(fused, naive)

    masks = torch.zeros((3, 8, 8), dtype=torch.bool)
    masks[0, 1:5, 1:5] = True
    masks[1, 1:5, 1:5] = True
    masks[2, 1:5, 1:5] = True
    scores = torch.tensor([0.9, 0.6, 0.7], dtype=torch.float32)
    labels = torch.tensor([1, 1, 2], dtype=torch.long)
    keep = class_wise_mask_nms(masks, scores, labels, iou_threshold=0.5)
    assert keep == [0, 2]


def test_closed_loop_forward_returns_stable_training_eval_and_inference_dicts() -> None:
    model = _build_test_model(num_queries=4)
    with torch.no_grad():
        model.query_head.class_embed.weight.zero_()
        model.query_head.class_embed.bias.zero_()
        model.query_head.class_embed.bias[1] = 6.0

    image = torch.rand(1, 3, 32, 40)
    outputs = model(
        image,
        targets=[_gt_instances()],
        processed_sizes=[(64, 64)],
        prompt_source="gt",
        gt_ratio=1.0,
        generator=torch.Generator().manual_seed(5),
    )

    assert set(outputs.keys()) == {"training_dict", "eval_dict", "inference_dict"}
    assert set(outputs["training_dict"].keys()) == {
        "coarse_outputs",
        "prompt_batches",
        "coarse_instance_batches",
        "refine_batches",
    }
    assert set(outputs["eval_dict"].keys()) == {"fused_batches", "instances"}
    assert set(outputs["inference_dict"].keys()) == {"instances"}

    fused_batch = outputs["eval_dict"]["fused_batches"][0]
    assert "instance_scores" in fused_batch
    assert torch.isfinite(fused_batch["instance_scores"]).all()
    assert outputs["inference_dict"]["instances"]
