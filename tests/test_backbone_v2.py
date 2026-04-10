"""Tests for SAM backbone v2 and LoRA wiring."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from models.sam_lora import QKVLoRALinear, load_lora_weights, save_lora_weights
from models.sam_backbone_v2 import SAMBackboneV2Config
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
    )


def test_lora_injects_only_last_six_attention_qkv_modules() -> None:
    model = _build_test_model()

    for block_index, block in enumerate(model.image_encoder.blocks):
        if block_index < 6:
            assert isinstance(block.attn.qkv, nn.Linear)
        else:
            assert isinstance(block.attn.qkv, QKVLoRALinear)
        assert not isinstance(block.mlp, QKVLoRALinear)


def test_lora_weights_round_trip(tmp_path: Path) -> None:
    model = _build_test_model()
    reference_model = _build_test_model()

    for _, block in enumerate(model.image_encoder.blocks[6:]):
        qkv = block.attn.qkv
        assert isinstance(qkv, QKVLoRALinear)
        qkv.lora_q_b.weight.data.fill_(0.25)
        qkv.lora_v_b.weight.data.fill_(0.5)

    checkpoint_path = tmp_path / "sam_lora.pth"
    save_lora_weights(model.image_encoder, checkpoint_path)
    load_lora_weights(reference_model.image_encoder, checkpoint_path)

    for block_index in range(6, 12):
        loaded = reference_model.image_encoder.blocks[block_index].attn.qkv
        assert isinstance(loaded, QKVLoRALinear)
        assert torch.allclose(loaded.lora_q_b.weight, torch.full_like(loaded.lora_q_b.weight, 0.25))
        assert torch.allclose(loaded.lora_v_b.weight, torch.full_like(loaded.lora_v_b.weight, 0.5))


def test_qkv_lora_keeps_key_branch_identical_to_base_projection() -> None:
    torch.manual_seed(0)
    base_linear = nn.Linear(8, 24, bias=False)
    lora_linear = QKVLoRALinear(base_linear, rank=2, alpha=4, dropout=0.0)
    lora_linear.lora_q_b.weight.data.fill_(0.25)
    lora_linear.lora_v_b.weight.data.fill_(0.5)
    inputs = torch.randn(2, 3, 8)

    base_output = base_linear(inputs)
    lora_output = lora_linear(inputs)
    embed_dim = lora_linear.embed_dim

    assert torch.allclose(lora_output[..., embed_dim : 2 * embed_dim], base_output[..., embed_dim : 2 * embed_dim])
    assert not torch.allclose(lora_output[..., :embed_dim], base_output[..., :embed_dim])
    assert not torch.allclose(lora_output[..., 2 * embed_dim :], base_output[..., 2 * embed_dim :])


def test_preprocess_scales_unit_range_to_sam_pixel_space() -> None:
    model = _build_test_model()
    image = torch.full((1, 3, 32, 40), 0.5, dtype=torch.float32)

    processed = model.preprocess_image(image)

    expected_channel0 = ((0.5 * 255.0) - 123.675) / 58.395
    assert processed.shape == (1, 3, 64, 64)
    assert torch.isclose(processed[0, 0, 0, 0], torch.tensor(expected_channel0), atol=1e-5)


def test_forward_backbone_outputs_multiscale_features() -> None:
    model = _build_test_model()
    image = torch.rand(2, 3, 32, 40)

    outputs = model.forward_backbone(image)

    assert set(outputs.keys()) == {
        "c2",
        "c3",
        "c4",
        "c5",
        "image_embeddings",
        "early_vit_feats",
    }
    for key in ("c2", "c3", "c4", "c5"):
        assert outputs[key].shape == (2, 64, 4, 4)
    assert outputs["image_embeddings"].shape == (2, 32, 4, 4)

    early_vit_feats = outputs["early_vit_feats"]
    assert isinstance(early_vit_feats, dict)
    assert set(early_vit_feats.keys()) == {"c2", "c3", "c4", "c5"}
    assert early_vit_feats["c2"].shape == (2, 64, 4, 4)
    for key in ("c2", "c3", "c4", "c5"):
        assert torch.equal(outputs[key], early_vit_feats[key])
    assert not hasattr(model, "feature_projections")


def test_top_level_model_registers_sam_modules_and_parameter_groups() -> None:
    model = _build_test_model()

    child_modules = dict(model.named_children())
    assert "image_encoder" in child_modules
    assert "prompt_encoder" in child_modules
    assert "hq_mask_decoder" in child_modules

    groups = model.get_parameter_groups()
    assert set(groups.keys()) == {
        "lora",
        "wirecr_adapter",
        "pixel_decoder",
        "query_head",
        "prompt_encoder",
        "hq_decoder",
    }
    assert groups["lora"]
    assert groups["wirecr_adapter"]
    assert groups["pixel_decoder"]
    assert groups["query_head"]
    assert groups["prompt_encoder"] == []
    assert groups["hq_decoder"]

    report = model.get_trainable_parameter_report()
    assert report["lora"]["parameter_count"] > 0
    assert report["wirecr_adapter"]["parameter_count"] > 0
    assert report["pixel_decoder"]["parameter_count"] > 0
    assert report["query_head"]["parameter_count"] > 0
    assert report["prompt_encoder"]["parameter_count"] == 0
    assert report["hq_decoder"]["parameter_count"] > 0
    assert report["summary"]["parameter_count"] == (
        report["lora"]["parameter_count"]
        + report["wirecr_adapter"]["parameter_count"]
        + report["pixel_decoder"]["parameter_count"]
        + report["query_head"]["parameter_count"]
        + report["hq_decoder"]["parameter_count"]
    )
    total_requires_grad = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total_requires_grad == report["summary"]["parameter_count"]

    state_dict = model.state_dict()
    assert "image_encoder.blocks.6.attn.qkv.lora_q_a.weight" in state_dict
    assert "prompt_encoder.no_mask_embed.weight" in state_dict
    assert "hq_mask_decoder.iou_token.weight" in state_dict
    assert "quality_head.score_mlp.0.weight" in state_dict
    assert "score_fusion.mlp.0.weight" in state_dict


def test_yaml_model_config_alias_maps_to_model_type_and_drives_model_build() -> None:
    config_path = Path("configs/wirecr_hqinstsam_vitb.yaml")
    model_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]

    resolved = SAMBackboneV2Config.from_model_config(
        model_config,
        image_size=64,
        prompt_embed_dim=32,
        feature_dim=32,
        encoder_embed_dim=64,
        encoder_depth=12,
        encoder_num_heads=4,
        encoder_global_attn_indexes=(2, 5, 8, 11),
    )
    assert resolved.model_type == "vit_b"
    assert resolved.lora_rank == model_config["lora_rank"]
    assert resolved.lora_alpha == model_config["lora_alpha"]
    assert resolved.lora_dropout == model_config["lora_dropout"]

    model = WireCRHQInstSAM.from_model_config(
        model_config,
        image_size=64,
        prompt_embed_dim=32,
        feature_dim=32,
        encoder_embed_dim=64,
        encoder_depth=12,
        encoder_num_heads=4,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        checkpoint=None,
    )
    report = model.get_trainable_parameter_report()
    assert report["lora"]["parameter_count"] > 0
