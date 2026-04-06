"""SAM backbone v2 with explicit module registration for the new mainline."""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib.util
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_SAM_MODELING_INIT = Path(__file__).resolve().parents[1] / "third_party" / "sam" / "segment_anything" / "modeling" / "__init__.py"


def _load_sam_modeling_module():
    module_name = "wirecr_sam_modeling"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        _SAM_MODELING_INIT,
        submodule_search_locations=[str(_SAM_MODELING_INIT.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load SAM modeling module from {_SAM_MODELING_INIT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SAM_MODELING = _load_sam_modeling_module()
ImageEncoderViT = _SAM_MODELING.ImageEncoderViT
MaskDecoder = _SAM_MODELING.MaskDecoder
PromptEncoder = _SAM_MODELING.PromptEncoder
Sam = _SAM_MODELING.Sam
TwoWayTransformer = _SAM_MODELING.TwoWayTransformer

from .sam_lora import LoRAInjectionSummary, get_lora_trainable_stats, inject_sam_lora

__all__ = [
    "SAMBackboneV2",
    "SAMBackboneV2Config",
    "build_sam_v2_modules",
    "normalize_sam_backbone_v2_config",
]


@dataclass(frozen=True)
class SAMBackboneV2Config:
    """Configuration for building a SAM backbone variant."""

    model_type: str = "vit_b"
    checkpoint: str | None = None
    image_size: int = 1024
    prompt_embed_dim: int = 256
    feature_dim: int = 256
    encoder_embed_dim: int | None = None
    encoder_depth: int | None = None
    encoder_num_heads: int | None = None
    encoder_global_attn_indexes: Tuple[int, ...] | None = None
    out_indices: Tuple[int, int, int, int] | None = None
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_num_last_blocks: int = 6
    enable_lora: bool = True
    freeze_prompt_encoder: bool = True
    freeze_hq_decoder: bool = True

    @classmethod
    def from_model_config(cls, model_config: Dict[str, object] | None = None, **overrides) -> "SAMBackboneV2Config":
        normalized = normalize_sam_backbone_v2_config(model_config or {}, **overrides)
        return cls(**normalized)


_MODEL_SPECS = {
    "vit_b": {
        "encoder_embed_dim": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "encoder_global_attn_indexes": (2, 5, 8, 11),
    },
    "vit_l": {
        "encoder_embed_dim": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": (5, 11, 17, 23),
    },
}


def _default_out_indices(depth: int) -> Tuple[int, int, int, int]:
    defaults = {
        12: (2, 5, 8, 11),
        24: (5, 11, 17, 23),
        32: (7, 15, 23, 31),
    }
    if depth not in defaults:
        raise ValueError(f"Unsupported encoder depth for multiscale extraction: {depth}")
    return defaults[depth]


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def normalize_sam_backbone_v2_config(
    model_config: Dict[str, object] | None = None,
    **overrides,
) -> Dict[str, object]:
    """Normalize YAML-facing config keys to SAMBackboneV2Config kwargs."""

    raw_config = dict(model_config or {})
    alias_model_type = raw_config.pop("sam_model_type", None)
    explicit_model_type = raw_config.get("model_type")
    if alias_model_type is not None and explicit_model_type is not None and alias_model_type != explicit_model_type:
        raise ValueError(
            f"Conflicting SAM model type keys: sam_model_type={alias_model_type!r}, "
            f"model_type={explicit_model_type!r}"
        )
    if explicit_model_type is None and alias_model_type is not None:
        raw_config["model_type"] = alias_model_type

    raw_config.update(overrides)
    supported_keys = {field.name for field in fields(SAMBackboneV2Config)}
    return {key: value for key, value in raw_config.items() if key in supported_keys}


def _resolve_config(config: SAMBackboneV2Config) -> dict[str, object]:
    if config.model_type not in _MODEL_SPECS:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    spec = dict(_MODEL_SPECS[config.model_type])
    if config.encoder_embed_dim is not None:
        spec["encoder_embed_dim"] = config.encoder_embed_dim
    if config.encoder_depth is not None:
        spec["encoder_depth"] = config.encoder_depth
    if config.encoder_num_heads is not None:
        spec["encoder_num_heads"] = config.encoder_num_heads
    if config.encoder_global_attn_indexes is not None:
        spec["encoder_global_attn_indexes"] = tuple(config.encoder_global_attn_indexes)
    spec["out_indices"] = tuple(config.out_indices or _default_out_indices(spec["encoder_depth"]))
    return spec


def build_sam_v2_modules(config: SAMBackboneV2Config) -> Sam:
    """Build a SAM model with configurable image size for testing and training."""

    spec = _resolve_config(config)
    image_embedding_size = config.image_size // 16
    sam_model = Sam(
        image_encoder=ImageEncoderViT(
            depth=spec["encoder_depth"],
            embed_dim=spec["encoder_embed_dim"],
            img_size=config.image_size,
            mlp_ratio=4,
            num_heads=spec["encoder_num_heads"],
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=spec["encoder_global_attn_indexes"],
            window_size=14,
            out_chans=config.prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=config.prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(config.image_size, config.image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=config.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=config.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if config.checkpoint is not None:
        state_dict = torch.load(config.checkpoint, map_location="cpu")
        sam_model.load_state_dict(state_dict)

    return sam_model


class SAMBackboneV2(nn.Module):
    """SAM backbone wrapper exposing multiscale features and explicit submodules."""

    feature_names = ("c2", "c3", "c4", "c5")

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = SAMBackboneV2Config(**kwargs)
        resolved = _resolve_config(self.config)
        self.model_type = self.config.model_type
        self.out_indices = tuple(resolved["out_indices"])

        sam_model = build_sam_v2_modules(self.config)
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.hq_mask_decoder = sam_model.mask_decoder
        self.embed_dim = resolved["encoder_embed_dim"]
        self.feature_dim = self.embed_dim
        self.image_size = self.config.image_size

        self.register_buffer("pixel_mean", sam_model.pixel_mean.detach().clone(), persistent=False)
        self.register_buffer("pixel_std", sam_model.pixel_std.detach().clone(), persistent=False)

        self.lora_summary: LoRAInjectionSummary | None = None
        if self.config.enable_lora:
            self.lora_summary = inject_sam_lora(
                self.image_encoder,
                num_last_blocks=self.config.lora_num_last_blocks,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )

        if self.config.freeze_prompt_encoder:
            _freeze_module(self.prompt_encoder)
        if self.config.freeze_hq_decoder:
            _freeze_module(self.hq_mask_decoder)

    def preprocess_image(self, images: torch.Tensor) -> torch.Tensor:
        """Scale [0, 1] floats to [0, 255], normalize, and pad to SAM size."""

        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.float()
        if images.is_floating_point():
            max_value = float(images.detach().amax().item()) if images.numel() else 0.0
            min_value = float(images.detach().amin().item()) if images.numel() else 0.0
            if min_value >= 0.0 and max_value <= 1.0:
                images = images * 255.0
        images = (images - self.pixel_mean) / self.pixel_std

        height, width = images.shape[-2:]
        pad_h = self.image_encoder.img_size - height
        pad_w = self.image_encoder.img_size - width
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Input image {height}x{width} exceeds configured SAM size {self.image_encoder.img_size}"
            )
        return F.pad(images, (0, pad_w, 0, pad_h))

    def _tokens_to_bchw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4 and tensor.shape[-1] == self.embed_dim:
            return tensor.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unsupported feature shape: {tuple(tensor.shape)}")

    def forward_backbone(self, images: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """Run SAM image encoder and expose multiscale features."""

        x = self.preprocess_image(images)
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        raw_features: Dict[str, torch.Tensor] = {}
        requested = set(self.out_indices)
        for block_index, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            if block_index in requested:
                feature_name = self.feature_names[self.out_indices.index(block_index)]
                raw_features[feature_name] = self._tokens_to_bchw(x)

        if len(raw_features) != 4:
            missing = [name for name in self.feature_names if name not in raw_features]
            raise RuntimeError(f"Failed to collect all multiscale SAM features: {missing}")

        image_embeddings = self.image_encoder.neck(self._tokens_to_bchw(x))

        return {
            **raw_features,
            "image_embeddings": image_embeddings,
            "early_vit_feats": dict(raw_features),
        }

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        return self.forward_backbone(images)

    def iter_lora_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if ".lora_" in name and parameter.requires_grad:
                yield parameter

    def get_lora_report(self) -> Dict[str, object]:
        return get_lora_trainable_stats(self.image_encoder)
