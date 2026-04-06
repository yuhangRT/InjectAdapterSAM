"""Stable instance transforms for the S02 dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

try:  # Pillow compatibility across versions
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    RESAMPLE_NEAREST = Image.Resampling.NEAREST  # type: ignore[attr-defined]
    ROTATE_RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    ROTATE_RESAMPLE_NEAREST = Image.Resampling.NEAREST  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST
    ROTATE_RESAMPLE_BILINEAR = Image.BILINEAR
    ROTATE_RESAMPLE_NEAREST = Image.NEAREST

__all__ = [
    "StableAugmentConfig",
    "StableInstanceTransforms",
    "build_stable_transforms",
]


@dataclass(frozen=True)
class StableAugmentConfig:
    image_size: int = 1024
    rotate_prob: float = 0.2
    rotate_deg: float = 2.0
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    blur_prob: float = 0.15
    blur_radius: tuple[float, float] = (0.6, 1.0)
    jpeg_prob: float = 0.2
    jpeg_quality: tuple[int, int] = (75, 95)
    noise_prob: float = 0.15
    noise_std: tuple[float, float] = (0.0, 0.01)
    pad_fill: tuple[int, int, int] = (0, 0, 0)


def _to_uint8_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _maybe_apply_rotation(
    image: Image.Image,
    masks: list[np.ndarray],
    rng: np.random.Generator,
    rotate_prob: float,
    rotate_deg: float,
) -> tuple[Image.Image, list[np.ndarray]]:
    if rotate_prob <= 0 or rotate_deg <= 0 or rng.random() >= rotate_prob:
        return image, masks
    angle = float(rng.uniform(-rotate_deg, rotate_deg))
    rotated_image = image.rotate(angle, resample=ROTATE_RESAMPLE_BILINEAR, expand=False, fillcolor=(0, 0, 0))
    rotated_masks = [
        np.asarray(_mask_to_image(mask).rotate(angle, resample=ROTATE_RESAMPLE_NEAREST, expand=False, fillcolor=0), dtype=np.uint8)
        for mask in masks
    ]
    return rotated_image, rotated_masks


def _maybe_apply_photometric(
    image: Image.Image,
    rng: np.random.Generator,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
    blur_prob: float,
    blur_radius: tuple[float, float],
    jpeg_prob: float,
    jpeg_quality: tuple[int, int],
    noise_prob: float,
    noise_std: tuple[float, float],
) -> Image.Image:
    result = image
    if brightness_range[0] != 1.0 or brightness_range[1] != 1.0:
        factor = float(rng.uniform(*brightness_range))
        result = ImageEnhance.Brightness(result).enhance(factor)
    if contrast_range[0] != 1.0 or contrast_range[1] != 1.0:
        factor = float(rng.uniform(*contrast_range))
        result = ImageEnhance.Contrast(result).enhance(factor)
    if blur_prob > 0 and rng.random() < blur_prob:
        radius = float(rng.uniform(*blur_radius))
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))
    if noise_prob > 0 and rng.random() < noise_prob:
        array = np.asarray(result, dtype=np.float32) / 255.0
        noise_sigma = float(rng.uniform(*noise_std))
        if noise_sigma > 0:
            array = np.clip(array + rng.normal(0.0, noise_sigma, size=array.shape), 0.0, 1.0)
            result = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
    if jpeg_prob > 0 and rng.random() < jpeg_prob:
        quality = int(round(float(rng.uniform(*jpeg_quality))))
        buffer = BytesIO()
        result.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB").copy()
    return result


def _resize_and_pad(
    image: Image.Image,
    masks: list[np.ndarray],
    image_size: int,
    pad_fill: tuple[int, int, int],
) -> tuple[Image.Image, list[np.ndarray], dict[str, Any]]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    scale = min(image_size / float(width), image_size / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    left = max((image_size - resized_width) // 2, 0)
    top = max((image_size - resized_height) // 2, 0)

    resized_image = image.resize((resized_width, resized_height), resample=RESAMPLE_BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), pad_fill)
    canvas.paste(resized_image, (left, top))

    transformed_masks: list[np.ndarray] = []
    for mask in masks:
        mask_image = _mask_to_image(mask)
        resized_mask = mask_image.resize((resized_width, resized_height), resample=RESAMPLE_NEAREST)
        mask_canvas = Image.new("L", (image_size, image_size), 0)
        mask_canvas.paste(resized_mask, (left, top))
        transformed_masks.append(np.asarray(mask_canvas, dtype=np.uint8))

    return canvas, transformed_masks, {
        "scale": float(scale),
        "resized_size": (int(resized_height), int(resized_width)),
        "pad": (int(left), int(top)),
    }


class StableInstanceTransforms:
    """Stable image/mask augmentations for the S02 data pipeline."""

    def __init__(self, config: StableAugmentConfig | None = None) -> None:
        self.config = config or StableAugmentConfig()

    def __call__(
        self,
        image: Image.Image,
        masks: list[np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        image = _to_uint8_image(image)
        masks = [np.asarray(mask, dtype=np.uint8) for mask in masks]
        image, masks = _maybe_apply_rotation(
            image=image,
            masks=masks,
            rng=rng,
            rotate_prob=self.config.rotate_prob,
            rotate_deg=self.config.rotate_deg,
        )
        image = _maybe_apply_photometric(
            image=image,
            rng=rng,
            brightness_range=self.config.brightness_range,
            contrast_range=self.config.contrast_range,
            blur_prob=self.config.blur_prob,
            blur_radius=self.config.blur_radius,
            jpeg_prob=self.config.jpeg_prob,
            jpeg_quality=self.config.jpeg_quality,
            noise_prob=self.config.noise_prob,
            noise_std=self.config.noise_std,
        )
        image, masks, resize_meta = _resize_and_pad(
            image=image,
            masks=masks,
            image_size=self.config.image_size,
            pad_fill=self.config.pad_fill,
        )
        image_tensor = _image_to_tensor(image)
        if masks:
            mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
        else:
            mask_tensor = torch.zeros((0, self.config.image_size, self.config.image_size), dtype=torch.uint8)
        return image_tensor, mask_tensor, resize_meta


def build_stable_transforms(**kwargs: Any) -> StableInstanceTransforms:
    return StableInstanceTransforms(StableAugmentConfig(**kwargs))
