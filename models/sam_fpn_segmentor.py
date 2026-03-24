"""
WireCR-SAM FPN semantic segmentation head.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crnet_adapter import ADAPTER_CONFIGS, crnet_adapter
from .crnet_blocks import ConvBN
from .decoder_fpn import WireCRFPNDecoder
from .sam_backbone_multi import SAMMultiFeatureBackbone

__all__ = ["SAMWithCRNetFPN", "create_sam_with_fpn"]


class SAMWithCRNetFPN(nn.Module):
    """Parallel FPN-based semantic segmentation head on top of the SAM encoder."""

    FEATURE_LEVELS = ("c2", "c3", "c4", "c5")
    DEFAULT_ADAPTER_LEVELS = ("c4", "c5")
    VALID_COMPRESSION_RATIOS = {4, 8, 16, 32, 64}
    LOW_LEVEL_LIGHTWEIGHT_DEFAULTS = {
        "adapter_size": "small",
        "compression_ratio": 16,
        "simple": True,
    }

    def __init__(
        self,
        sam_model,
        adapter_config,
        num_classes: int = 3,
        class_names=None,
        disable_adapter: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        if num_classes != 3:
            raise ValueError("SAMWithCRNetFPN currently supports exactly 3 semantic classes.")

        self.sam_model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.num_classes = num_classes
        self.class_names = list(class_names or ["background", "wire", "interface-hole"])
        self.disable_adapter = disable_adapter
        self.freeze_encoder = freeze_encoder
        self.adapter_config = dict(adapter_config)
        self.feature_channels = 256
        self.debug_feature_names = self.FEATURE_LEVELS

        self.backbone = SAMMultiFeatureBackbone(self.image_encoder)
        embed_dim = self.backbone.embed_dim

        for level_name in self.FEATURE_LEVELS:
            setattr(self, f"proj_{level_name}", ConvBN(embed_dim, self.feature_channels, 1))

        self.enabled_adapter_levels, self.level_adapter_specs = self._resolve_level_adapter_specs()
        for level_name in self.FEATURE_LEVELS:
            setattr(self, f"adapter_{level_name}", self._build_level_adapter(level_name))

        self.decoder = WireCRFPNDecoder(in_channels=self.feature_channels, fpn_channels=128, num_classes=num_classes)
        self.hole_aux_head = nn.Sequential(
            ConvBN(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        if freeze_encoder:
            self._freeze_module(self.image_encoder)

    def _freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    @classmethod
    def _parse_feature_levels(cls, raw_levels) -> tuple[str, ...]:
        if raw_levels is None:
            return cls.DEFAULT_ADAPTER_LEVELS

        if isinstance(raw_levels, str):
            tokens = [token.strip().lower() for token in raw_levels.split(",") if token.strip()]
        else:
            tokens = [str(token).strip().lower() for token in raw_levels if str(token).strip()]

        if not tokens:
            return ()

        normalized = []
        for level_name in tokens:
            if level_name not in cls.FEATURE_LEVELS:
                raise ValueError(
                    f"Unsupported FPN feature level '{level_name}'. Expected one of: {', '.join(cls.FEATURE_LEVELS)}"
                )
            if level_name not in normalized:
                normalized.append(level_name)
        return tuple(normalized)

    @staticmethod
    def _parse_bool_token(raw_value: str) -> bool:
        value = str(raw_value).strip().lower()
        truthy = {"1", "true", "t", "yes", "y", "on", "simple"}
        falsy = {"0", "false", "f", "no", "n", "off", "full"}
        if value in truthy:
            return True
        if value in falsy:
            return False
        raise ValueError(
            f"Unsupported boolean token '{raw_value}'. Use one of: 1/0, true/false, yes/no, simple/full."
        )

    @classmethod
    def _parse_level_overrides(cls, raw_value, value_parser, field_name: str) -> dict[str, object]:
        if raw_value is None:
            return {}

        items = [item.strip() for item in str(raw_value).split(",") if item.strip()]
        if not items:
            return {}

        overrides = {}
        for item in items:
            if "=" not in item:
                raise ValueError(f"Invalid {field_name} item '{item}'. Expected LEVEL=VALUE.")
            level_name, value = item.split("=", 1)
            level_name = level_name.strip().lower()
            if level_name not in cls.FEATURE_LEVELS:
                raise ValueError(
                    f"Unsupported level '{level_name}' in {field_name}. Expected one of: {', '.join(cls.FEATURE_LEVELS)}"
                )
            overrides[level_name] = value_parser(value.strip())
        return overrides

    @classmethod
    def _validate_adapter_size(cls, adapter_size: str) -> str:
        adapter_size = str(adapter_size).strip().lower()
        if adapter_size not in ADAPTER_CONFIGS:
            raise ValueError(
                f"Invalid adapter size '{adapter_size}'. Expected one of: {', '.join(sorted(ADAPTER_CONFIGS))}"
            )
        return adapter_size

    @classmethod
    def _validate_compression_ratio(cls, compression_ratio) -> int:
        value = int(compression_ratio)
        if value not in cls.VALID_COMPRESSION_RATIOS:
            raise ValueError(
                f"Invalid compression ratio '{compression_ratio}'. "
                f"Expected one of: {', '.join(str(item) for item in sorted(cls.VALID_COMPRESSION_RATIOS))}"
            )
        return value

    @staticmethod
    def _validate_adapter_kind(adapter_kind) -> str:
        value = str(adapter_kind).strip().lower()
        if value not in {"wirecr", "vanilla"}:
            raise ValueError(f"Invalid adapter kind '{adapter_kind}'. Expected one of: wirecr, vanilla")
        return value

    def _resolve_level_adapter_specs(self) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
        enabled_levels = self._parse_feature_levels(
            self.adapter_config.get("fpn_adapter_levels", self.DEFAULT_ADAPTER_LEVELS)
        )
        base_spec = {
            level_name: {
                "adapter_kind": self._validate_adapter_kind(self.adapter_config.get("adapter_kind", "wirecr")),
                "adapter_size": self._validate_adapter_size(self.adapter_config.get("adapter_size", "medium")),
                "compression_ratio": self._validate_compression_ratio(
                    self.adapter_config.get("compression_ratio", 8)
                ),
                "use_residual": bool(self.adapter_config.get("use_residual", True)),
                "simple": bool(self.adapter_config.get("simple", False)),
            }
            for level_name in self.FEATURE_LEVELS
        }

        size_overrides = self._parse_level_overrides(
            self.adapter_config.get("fpn_adapter_size_map"),
            self._validate_adapter_size,
            "fpn_adapter_size_map",
        )
        compression_overrides = self._parse_level_overrides(
            self.adapter_config.get("fpn_compression_map"),
            self._validate_compression_ratio,
            "fpn_compression_map",
        )
        simple_overrides = self._parse_level_overrides(
            self.adapter_config.get("fpn_simple_map"),
            self._parse_bool_token,
            "fpn_simple_map",
        )

        for level_name in enabled_levels:
            if level_name in {"c2", "c3"}:
                if level_name not in size_overrides:
                    base_spec[level_name]["adapter_size"] = self.LOW_LEVEL_LIGHTWEIGHT_DEFAULTS["adapter_size"]
                if level_name not in compression_overrides:
                    base_spec[level_name]["compression_ratio"] = self.LOW_LEVEL_LIGHTWEIGHT_DEFAULTS["compression_ratio"]
                if level_name not in simple_overrides:
                    base_spec[level_name]["simple"] = self.LOW_LEVEL_LIGHTWEIGHT_DEFAULTS["simple"]

        for level_name, adapter_size in size_overrides.items():
            base_spec[level_name]["adapter_size"] = adapter_size
        for level_name, compression_ratio in compression_overrides.items():
            base_spec[level_name]["compression_ratio"] = compression_ratio
        for level_name, simple_flag in simple_overrides.items():
            base_spec[level_name]["simple"] = simple_flag

        return enabled_levels, base_spec

    def _projection_module(self, level_name: str) -> nn.Module:
        return getattr(self, f"proj_{level_name}")

    def _adapter_module(self, level_name: str) -> nn.Module:
        return getattr(self, f"adapter_{level_name}")

    def _build_level_adapter(self, level_name: str) -> nn.Module:
        if self.disable_adapter or level_name not in self.enabled_adapter_levels:
            return nn.Identity()

        level_spec = self.level_adapter_specs[level_name]
        return crnet_adapter(
            in_channels=self.feature_channels,
            adapter_kind=level_spec["adapter_kind"],
            adapter_size=level_spec["adapter_size"],
            compression_ratio=level_spec["compression_ratio"],
            use_residual=level_spec["use_residual"],
            simple=level_spec["simple"],
        )

    def _project_features(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            level_name: self._adapter_module(level_name)(self._projection_module(level_name)(features[level_name]))
            for level_name in self.FEATURE_LEVELS
        }

    @staticmethod
    def _target_output_size(batched_input: List[Dict[str, Any]]) -> tuple[int, int]:
        sizes = [tuple(record.get("output_size", record["image"].shape[-2:])) for record in batched_input]
        if any(size != sizes[0] for size in sizes[1:]):
            raise ValueError("SAMWithCRNetFPN expects a shared output_size across the batch.")
        return sizes[0]

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del multimask_output
        if not batched_input:
            raise ValueError("batched_input must not be empty")

        device = self.sam_model.device
        input_images = torch.stack(
            [self.sam_model.preprocess(record["image"].to(device).float()) for record in batched_input],
            dim=0,
        )

        raw_features = self.backbone(input_images)
        pyramid_features = self._project_features(raw_features)
        main_logits, fused_feat = self.decoder(pyramid_features)
        hole_aux_logits = self.hole_aux_head(fused_feat)

        output_size = self._target_output_size(batched_input)
        main_logits = F.interpolate(main_logits, size=output_size, mode="bilinear", align_corners=False)
        hole_aux_logits = F.interpolate(hole_aux_logits, size=output_size, mode="bilinear", align_corners=False)
        pred_masks = main_logits.argmax(dim=1)

        outputs = {
            "main_logits": main_logits,
            "hole_aux_logits": hole_aux_logits,
            "pred_masks": pred_masks,
        }
        if return_features:
            outputs["pyramid_features"] = pyramid_features
        return outputs

    def get_trainable_params(self):
        return [param for param in self.parameters() if param.requires_grad]

    def get_num_total_params(self):
        return sum(param.numel() for param in self.parameters())

    def get_num_frozen_params(self):
        return sum(param.numel() for param in self.parameters() if not param.requires_grad)

    def get_num_adapter_params(self):
        modules = [self.decoder, self.hole_aux_head]
        for level_name in self.FEATURE_LEVELS:
            modules.append(self._projection_module(level_name))
            modules.append(self._adapter_module(level_name))
        return sum(param.numel() for module in modules for param in module.parameters())

    def print_model_info(self):
        total_params = self.get_num_total_params()
        adapter_params = self.get_num_adapter_params()
        frozen_params = self.get_num_frozen_params()

        print(f"\n{'=' * 60}")
        print("WireCR-SAM FPN Model Info")
        print(f"{'=' * 60}")
        print(f"Head type: fpn")
        print(f"Adapter enabled: {not self.disable_adapter}")
        print(f"Adapter kind: {self.adapter_config.get('adapter_kind', 'wirecr')}")
        print(f"Adapter size: {self.adapter_config.get('adapter_size', 'medium')}")
        print(f"Compression ratio: 1/{self.adapter_config.get('compression_ratio', 8)}")
        print(f"Use residual: {self.adapter_config.get('use_residual', True)}")
        print(f"Adapter levels: {', '.join(self.enabled_adapter_levels) if self.enabled_adapter_levels else 'none'}")
        if not self.disable_adapter:
            print("Per-level adapter config:")
            for level_name in self.FEATURE_LEVELS:
                if level_name not in self.enabled_adapter_levels:
                    print(f"  {level_name}: off")
                    continue
                level_spec = self.level_adapter_specs[level_name]
                level_variant = "simple" if level_spec["simple"] else "full"
                print(
                    f"  {level_name}: kind={level_spec['adapter_kind']}, {level_variant}, size={level_spec['adapter_size']}, "
                    f"compression=1/{level_spec['compression_ratio']}"
                )
        print(f"Semantic classes: {self.num_classes}")
        print("\nParameter counts:")
        print(f"  Total params: {total_params:,}")
        print(f"  New-head params: {adapter_params:,} ({100 * adapter_params / total_params:.2f}%)")
        print(f"  Frozen params: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
        print(f"  Trainable params: {total_params - frozen_params:,}")
        print(f"{'=' * 60}\n")


def create_sam_with_fpn(
    sam_model,
    adapter_kind: str = "wirecr",
    adapter_size: str = "medium",
    compression_ratio: int = 8,
    use_residual: bool = True,
    simple: bool = False,
    num_classes: int = 3,
    disable_adapter: bool = False,
    freeze_encoder: bool = True,
):
    adapter_config = {
        "adapter_kind": adapter_kind,
        "adapter_size": adapter_size,
        "compression_ratio": compression_ratio,
        "use_residual": use_residual,
        "simple": simple,
    }
    return SAMWithCRNetFPN(
        sam_model=sam_model,
        adapter_config=adapter_config,
        num_classes=num_classes,
        disable_adapter=disable_adapter,
        freeze_encoder=freeze_encoder,
    )
