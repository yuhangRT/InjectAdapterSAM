"""
Backend abstraction for SAM-family image/prompt/mask interfaces.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SAMBackendConfig:
    backend_type: str = "auto"
    sam_model_type: str = "vit_b"
    sam_checkpoint: str | None = None
    fallback_sam_model_type: str = "vit_b"
    fallback_sam_checkpoint: str | None = None


class SAMBackendBase:
    backend_name = "sam-base"

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    def image_size(self) -> int:
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        raise NotImplementedError

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_dense_pe(self) -> torch.Tensor:
        raise NotImplementedError

    def encode_prompts(
        self,
        *,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def decode_masks(
        self,
        *,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SAM1Backend(SAMBackendBase):
    backend_name = "sam1"

    def __init__(self, sam_model) -> None:
        self.sam_model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder

    @property
    def device(self) -> torch.device:
        for module in (self.image_encoder, self.prompt_encoder, self.mask_decoder):
            try:
                return next(module.parameters()).device
            except StopIteration:
                continue
        return self.sam_model.device

    @property
    def image_size(self) -> int:
        return int(self.image_encoder.img_size)

    @property
    def embed_dim(self) -> int:
        return int(self.image_encoder.patch_embed.proj.out_channels)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        # The instance dataloader emits float images in [0, 1], while the
        # original SAM pixel mean/std are defined for 0-255 RGB inputs.
        # Keep already-unscaled inputs unchanged.
        if image.is_floating_point() and float(image.detach().amax().item()) <= 1.5:
            image = image * 255.0
        pixel_mean = self.sam_model.pixel_mean.to(device=image.device, dtype=image.dtype)
        pixel_std = self.sam_model.pixel_std.to(device=image.device, dtype=image.dtype)
        image = (image - pixel_mean) / pixel_std
        h, w = image.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        return torch.nn.functional.pad(image, (0, padw, 0, padh))

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        return self.sam_model.postprocess_masks(masks, input_size=input_size, original_size=original_size)

    def get_dense_pe(self) -> torch.Tensor:
        return self.prompt_encoder.get_dense_pe()

    def encode_prompts(
        self,
        *,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prompt_encoder(points=points, boxes=boxes, masks=masks)

    def decode_masks(
        self,
        *,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )


class SAM21Backend(SAMBackendBase):
    backend_name = "sam2.1"

    def __init__(self, *args, **kwargs) -> None:
        raise ImportError(
            "SAM2.1 backend support requires the official sam2 package in the environment. "
            "This repository currently falls back to the existing SAM1 integration when SAM2.1 is unavailable."
        )


def _build_sam1(model_type: str, checkpoint_path: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sam_repo = os.path.join(repo_root, "third_party", "sam")
    if sam_repo not in sys.path:
        sys.path.insert(0, sam_repo)
    from segment_anything import sam_model_registry

    return sam_model_registry[model_type](checkpoint=checkpoint_path)


def build_sam_backend(config: SAMBackendConfig) -> SAMBackendBase:
    backend_type = str(config.backend_type).strip().lower()
    checkpoint_path = config.sam_checkpoint
    if checkpoint_path is None:
        raise ValueError("sam_checkpoint is required to build a SAM backend.")
    if backend_type in {"auto", "sam2", "sam2.1", "hiera"}:
        try:
            return SAM21Backend(model_type=config.sam_model_type, checkpoint_path=checkpoint_path)
        except Exception:
            if backend_type not in {"auto"}:
                raise

    fallback_checkpoint = config.fallback_sam_checkpoint or checkpoint_path
    fallback_model_type = config.fallback_sam_model_type or config.sam_model_type
    try:
        sam_model = _build_sam1(fallback_model_type, fallback_checkpoint)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize the SAM1 fallback backend. "
            "If --sam-model-type points to a SAM2.1 checkpoint, also provide "
            "--fallback-sam-model-type and --fallback-sam-checkpoint for the existing SAM1 path."
        ) from exc
    return SAM1Backend(sam_model)
