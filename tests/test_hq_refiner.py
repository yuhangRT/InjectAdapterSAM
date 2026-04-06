"""Tests for the S07 HQ refiner and refine losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.quality_head import QualityHead
from models.wirecr_hq_instsam import WireCRHQInstSAM
from utils.losses_refine import RefineLossCriterion


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


def _mean_iou(logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    preds = logits.sigmoid() > 0.5
    targets = target_masks > 0.5
    intersection = (preds & targets).float().sum(dim=(1, 2, 3))
    union = (preds | targets).float().sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (intersection / union).mean()


def test_hq_refiner_gt_prompt_improves_over_decoder_coarse() -> None:
    model = _build_test_model()
    with torch.no_grad():
        model.query_head.class_embed.weight.zero_()
        model.query_head.class_embed.bias.zero_()
        model.query_head.class_embed.bias[1] = 6.0

    image = torch.rand(1, 3, 32, 40)
    targets = [_gt_instances()]
    coarse_outputs = model.forward_coarse(image)
    prompt_outputs = model.build_prompts(
        coarse_outputs,
        targets=targets,
        processed_sizes=[(64, 64)],
        prompt_source="gt",
        gt_ratio=1.0,
        generator=torch.Generator().manual_seed(3),
    )
    refine_outputs = model.forward_refine(
        image,
        coarse_outputs=coarse_outputs,
        prompt_batches=prompt_outputs["prompt_batches"],
        coarse_instance_batches=prompt_outputs["coarse_instance_batches"],
        targets=targets,
        processed_sizes=[(64, 64)],
        prompt_source="gt",
        gt_ratio=1.0,
        generator=torch.Generator().manual_seed(3),
    )

    batch = refine_outputs["refine_batches"][0]
    target_masks = F.interpolate(targets[0]["masks"].unsqueeze(1), size=(16, 16), mode="nearest")
    target_masks = F.interpolate(target_masks, size=batch["refined_mask_logits"].shape[-2:], mode="nearest")

    coarse_iou = _mean_iou(batch["coarse_mask_logits"], target_masks)
    refined_iou = _mean_iou(batch["refined_mask_logits"], target_masks)

    assert batch["refined_mask_logits"].shape == batch["coarse_mask_logits"].shape
    assert batch["quality_scores"].shape == (2,)
    assert refined_iou > coarse_iou


def test_quality_head_depends_on_features_not_simple_score_product() -> None:
    quality_head = QualityHead(in_channels=4, hidden_dim=8)
    refine_features = torch.stack(
        (
            torch.ones(4, 8, 8),
            torch.full((4, 8, 8), 2.0),
        ),
        dim=0,
    )
    coarse_score = torch.tensor([0.5, 0.5], dtype=torch.float32)
    decoder_score = torch.tensor([0.4, 0.4], dtype=torch.float32)

    quality = quality_head(
        refine_features,
        coarse_score=coarse_score,
        decoder_score=decoder_score,
    )

    naive = coarse_score * decoder_score
    assert torch.isfinite(quality).all()
    assert not torch.allclose(quality, naive)
    assert not torch.allclose(quality[0], quality[1])


def test_refine_losses_include_boundary_and_quality_terms() -> None:
    criterion = RefineLossCriterion()
    refined_mask_logits = torch.tensor(
        [
            [[[6.0, 6.0, -6.0, -6.0], [6.0, 6.0, -6.0, -6.0], [-6.0, -6.0, -6.0, -6.0], [-6.0, -6.0, -6.0, -6.0]]],
            [[[-6.0, -6.0, -6.0, -6.0], [-6.0, 6.0, 6.0, -6.0], [-6.0, 6.0, 6.0, -6.0], [-6.0, -6.0, -6.0, -6.0]]],
        ],
        dtype=torch.float32,
    )
    target_masks = torch.tensor(
        [
            [[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    quality_scores = torch.tensor([0.9, 0.2], dtype=torch.float32)

    losses = criterion(
        refined_mask_logits=refined_mask_logits,
        target_masks=target_masks,
        quality_scores=quality_scores,
    )

    assert torch.isfinite(losses["loss"])
    assert losses["loss_boundary"].item() > 0.0
    assert losses["loss_quality"].item() > 0.0
