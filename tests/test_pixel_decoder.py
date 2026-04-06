"""Tests for the S04 WireCR multi-scale adapter and pixel decoder."""

from __future__ import annotations

import torch

from models.crnet_adapter import WireCRAdapter, WireCRAdapterSimple
from models.pixel_decoder import WireCRPixelDecoder
from models.wirecr_hq_instsam import WireCRHQInstSAM
from models.wirecr_multiscale_adapter import WireCRMultiScaleAdapter


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
    )


def test_wirecr_multiscale_adapter_uses_lightweight_c2_path() -> None:
    adapter = WireCRMultiScaleAdapter(
        in_channels={"c2": 64, "c3": 64, "c4": 64, "c5": 64},
        out_channels=256,
    )
    features = {
        feature_name: torch.randn(2, 64, 4, 4)
        for feature_name in ("c2", "c3", "c4", "c5")
    }

    outputs = adapter(features)

    assert isinstance(adapter.adapters["c2"], WireCRAdapterSimple)
    assert isinstance(adapter.adapters["c3"], WireCRAdapter)
    assert isinstance(adapter.adapters["c4"], WireCRAdapter)
    assert isinstance(adapter.adapters["c5"], WireCRAdapter)
    for feature_name in ("c2", "c3", "c4", "c5"):
        assert outputs[feature_name].shape == (2, 256, 4, 4)


def test_pixel_decoder_outputs_stride4_mask_features_and_uses_c2() -> None:
    decoder = WireCRPixelDecoder(in_channels=256, out_channels=256)
    spatial_pattern = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32).reshape(1, 1, 4, 4)
    spatial_pattern = spatial_pattern.repeat(2, 256, 1, 1)
    zero_features = {
        feature_name: torch.zeros(2, 256, 4, 4, dtype=torch.float32)
        for feature_name in ("c2", "c3", "c4", "c5")
    }
    active_c2_features = dict(zero_features)
    active_c2_features["c2"] = spatial_pattern

    zero_outputs = decoder(zero_features)
    active_outputs = decoder(active_c2_features)

    assert zero_outputs["mask_features"].shape == (2, 256, 16, 16)
    assert active_outputs["mask_features"].shape == (2, 256, 16, 16)
    assert len(active_outputs["multi_scale_memory"]) == 4
    assert active_outputs["multi_scale_memory"][0].shape == (2, 256, 4, 4)
    assert not torch.allclose(zero_outputs["mask_features"], active_outputs["mask_features"])


def test_top_level_model_exposes_wirecr_and_pixel_decoder_groups() -> None:
    model = _build_test_model()
    image = torch.rand(2, 3, 32, 40)

    outputs = model.forward_pixel_decoder(image)
    groups = model.get_parameter_groups()

    adapted_features = outputs["adapted_features"]
    assert set(adapted_features.keys()) == {"c2", "c3", "c4", "c5"}
    for feature_name in ("c2", "c3", "c4", "c5"):
        assert adapted_features[feature_name].shape == (2, 256, 4, 4)

    assert outputs["mask_features"].shape == (2, 256, 16, 16)
    assert len(outputs["multi_scale_memory"]) == 4
    assert groups["wirecr_adapter"]
    assert groups["pixel_decoder"]
    assert groups["query_head"]
