"""Tests for the S05 coarse query head, matcher, and losses."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.matcher import HungarianMatcher
from models.query_instance_head import QueryInstanceHead
from models.wirecr_hq_instsam import WireCRHQInstSAM
from utils.losses_coarse import CoarseLossCriterion


def _build_test_model(*, num_queries: int = 64, query_decoder_layers: int = 6) -> WireCRHQInstSAM:
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
        query_decoder_layers=query_decoder_layers,
    )


def _make_square_mask(top: int, left: int, bottom: int, right: int, size: int = 16) -> torch.Tensor:
    mask = torch.zeros(size, size, dtype=torch.float32)
    mask[top:bottom, left:right] = 1.0
    return mask


def _make_target(
    *,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    processed_size: tuple[int, int] = (64, 64),
) -> dict[str, torch.Tensor | tuple[int, int]]:
    return {
        "labels": labels,
        "boxes": boxes,
        "masks": masks,
        "processed_size": processed_size,
    }


def test_query_head_defaults_and_config_override() -> None:
    default_model = _build_test_model()
    assert default_model.num_queries == 64
    assert default_model.query_decoder_layers == 6

    override_model = WireCRHQInstSAM.from_model_config(
        {
            "sam_model_type": "vit_b",
            "image_size": 64,
            "prompt_embed_dim": 32,
            "encoder_embed_dim": 64,
            "encoder_depth": 12,
            "encoder_num_heads": 4,
            "encoder_global_attn_indexes": (2, 5, 8, 11),
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "wirecr_out_channels": 256,
            "pixel_decoder_channels": 256,
            "num_queries": 100,
            "decoder_layers": 2,
        }
    )
    assert override_model.num_queries == 100
    assert override_model.query_decoder_layers == 2


def test_query_head_outputs_expected_shapes_and_groups() -> None:
    model = _build_test_model()
    image = torch.rand(1, 3, 32, 40)

    outputs = model.forward_query_head(image)
    groups = model.get_parameter_groups()

    assert outputs["pred_logits"].shape == (1, 64, 3)
    assert outputs["pred_boxes"].shape == (1, 64, 4)
    assert outputs["pred_masks"].shape == (1, 64, 16, 16)
    assert len(outputs["aux_outputs"]) == 5
    assert groups["query_head"]


def test_query_memory_includes_spatial_position_encoding() -> None:
    head = QueryInstanceHead(
        hidden_dim=32,
        num_classes=2,
        num_queries=4,
        decoder_layers=1,
        num_heads=4,
        dim_feedforward=64,
        num_feature_levels=2,
    )
    zero_features = [
        torch.zeros(1, 32, 2, 2, dtype=torch.float32),
        torch.zeros(1, 32, 2, 2, dtype=torch.float32),
    ]

    memory = head._build_memory(zero_features)

    assert memory.shape == (1, 8, 32)
    assert not torch.allclose(memory[0, 0], memory[0, 1])
    assert not torch.allclose(memory[0, 0], memory[0, 4])


def test_query_head_uses_distributed_reference_boxes_for_box_init() -> None:
    head = QueryInstanceHead(
        hidden_dim=32,
        num_classes=2,
        num_queries=9,
        decoder_layers=1,
        num_heads=4,
        dim_feedforward=64,
        num_feature_levels=2,
    )
    queries = torch.randn(1, 9, 32, dtype=torch.float32)
    mask_features = torch.randn(1, 32, 4, 4, dtype=torch.float32)

    outputs = head._predict(queries, mask_features)

    assert outputs["pred_boxes"].shape == (1, 9, 4)
    assert torch.allclose(outputs["pred_boxes"][0], head.reference_boxes, atol=1e-6)
    assert not torch.allclose(head.reference_boxes[0], head.reference_boxes[1])


def test_matcher_weights_and_empty_inputs() -> None:
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_mask_bce=5.0,
        cost_mask_dice=5.0,
    )
    assert matcher.cost_class == 2.0
    assert matcher.cost_bbox == 5.0
    assert matcher.cost_giou == 2.0
    assert matcher.cost_mask_bce == 5.0
    assert matcher.cost_mask_dice == 5.0

    empty_outputs = {
        "pred_logits": torch.zeros(1, 0, 3),
        "pred_boxes": torch.zeros(1, 0, 4),
        "pred_masks": torch.zeros(1, 0, 16, 16),
    }
    empty_targets = [
        _make_target(
            labels=torch.zeros((0,), dtype=torch.long),
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            masks=torch.zeros((0, 16, 16), dtype=torch.float32),
        )
    ]
    indices = matcher(empty_outputs, empty_targets)
    assert indices[0][0].numel() == 0
    assert indices[0][1].numel() == 0

    non_empty_outputs = {
        "pred_logits": torch.randn(1, 4, 3),
        "pred_boxes": torch.rand(1, 4, 4),
        "pred_masks": torch.randn(1, 4, 16, 16),
    }
    indices = matcher(non_empty_outputs, empty_targets)
    assert indices[0][0].numel() == 0
    assert indices[0][1].numel() == 0


def test_coarse_losses_backward_and_aux_accumulation() -> None:
    torch.manual_seed(0)
    model = _build_test_model()
    matcher = HungarianMatcher()
    criterion = CoarseLossCriterion(matcher=matcher)
    image = torch.rand(1, 3, 32, 40)
    outputs = model.forward_coarse(image)

    targets = [
        _make_target(
            labels=torch.tensor([1, 2], dtype=torch.long),
            boxes=torch.tensor(
                [
                    [4.0, 4.0, 32.0, 32.0],
                    [32.0, 32.0, 56.0, 56.0],
                ],
                dtype=torch.float32,
            ),
            masks=torch.stack(
                [
                    _make_square_mask(1, 1, 8, 8),
                    _make_square_mask(8, 8, 14, 14),
                ],
                dim=0,
            ),
        )
    ]

    losses = criterion(outputs, targets)

    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["loss_aux_total"])
    assert losses["loss_aux_total"].item() > 0.0

    losses["loss"].backward()
    assert model.query_head.class_embed.weight.grad is not None


def test_repulsion_loss_is_positive_for_adjacent_label_sleeve_instances() -> None:
    matcher = HungarianMatcher()
    criterion = CoarseLossCriterion(matcher=matcher, aux_loss_weight=0.0)

    pred_masks = torch.full((1, 2, 16, 16), -6.0)
    pred_masks[0, 0, 2:10, 2:10] = 6.0
    pred_masks[0, 1, 3:11, 5:13] = 6.0

    outputs = {
        "pred_logits": torch.tensor([[[0.0, 8.0, -8.0], [0.0, 7.5, -7.5]]], dtype=torch.float32),
        "pred_boxes": torch.tensor(
            [[[0.10, 0.10, 0.60, 0.60], [0.22, 0.12, 0.72, 0.62]]],
            dtype=torch.float32,
        ),
        "pred_masks": pred_masks,
        "aux_outputs": [],
    }
    targets = [
        _make_target(
            labels=torch.tensor([1, 1], dtype=torch.long),
            boxes=torch.tensor(
                [
                    [8.0, 8.0, 40.0, 40.0],
                    [20.0, 12.0, 52.0, 44.0],
                ],
                dtype=torch.float32,
            ),
            masks=torch.stack(
                [
                    _make_square_mask(2, 2, 10, 10),
                    _make_square_mask(3, 5, 11, 13),
                ],
                dim=0,
            ),
        )
    ]

    losses = criterion(outputs, targets)
    assert losses["loss_repulsion"].item() > 0.0


def test_target_box_normalization_uses_processed_size() -> None:
    matcher = HungarianMatcher()
    criterion = CoarseLossCriterion(matcher=matcher, aux_loss_weight=0.0)

    outputs = {
        "pred_logits": torch.tensor([[[0.0, 8.0, -8.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.25, 0.25, 0.75, 0.75]]], dtype=torch.float32),
        "pred_masks": _make_square_mask(4, 4, 12, 12).view(1, 1, 16, 16) * 12.0 - 6.0,
        "aux_outputs": [],
    }
    targets = [
        _make_target(
            labels=torch.tensor([1], dtype=torch.long),
            boxes=torch.tensor([[16.0, 16.0, 48.0, 48.0]], dtype=torch.float32),
            masks=_make_square_mask(4, 4, 12, 12).unsqueeze(0),
            processed_size=(64, 64),
        )
    ]

    losses = criterion(outputs, targets)
    assert losses["loss_bbox"].item() < 1e-6
    assert losses["loss_giou"].item() < 1e-5


def test_losses_module_imports_after_model_import() -> None:
    losses_module = importlib.import_module("utils.losses_coarse")
    assert losses_module.CoarseLossCriterion is CoarseLossCriterion


def test_coarse_losses_handle_empty_gt() -> None:
    matcher = HungarianMatcher()
    criterion = CoarseLossCriterion(matcher=matcher, aux_loss_weight=0.0)
    model = _build_test_model()
    outputs = model.forward_coarse(torch.rand(1, 3, 32, 40))
    empty_targets = [
        _make_target(
            labels=torch.zeros((0,), dtype=torch.long),
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            masks=torch.zeros((0, 16, 16), dtype=torch.float32),
        )
    ]

    losses = criterion(outputs, empty_targets)
    assert torch.isfinite(losses["loss"])
    assert losses["loss_bbox"].item() == 0.0
    assert losses["loss_giou"].item() == 0.0
