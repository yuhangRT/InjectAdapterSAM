"""
Industrial-oriented image augmentation for wire_hole segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover - optional in environments without opencv
    cv2 = None


__all__ = [
    "IndustrialAugmentor",
    "build_train_augmentor",
]


def _find_perspective_coeffs(src_points, dst_points) -> Tuple[float, ...]:
    matrix = []
    for (sx, sy), (dx, dy) in zip(src_points, dst_points):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(src_points, dtype=np.float64).reshape(8)
    coeffs = np.linalg.solve(a, b)
    return tuple(float(value) for value in coeffs)


def _soft_mask_from_draw(draw_fn, size: Tuple[int, int], blur_radius: float) -> np.ndarray:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw_fn(draw)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(mask, dtype=np.float32) / 255.0


def _clip_uint8(image_array: np.ndarray) -> np.ndarray:
    return np.clip(image_array, 0, 255).astype(np.uint8)


def _blend_local_effect(base_array: np.ndarray, target_array: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha_mask[..., None], 0.0, 1.0)
    return _clip_uint8(base_array * (1.0 - alpha) + target_array * alpha)


def _sample_center_bias(rng: np.random.Generator, length: int, margin_ratio: float = 0.08) -> float:
    if length <= 1:
        return 0.0
    margin = margin_ratio * length
    low = margin
    high = max(low + 1.0, length - margin)
    value = rng.beta(3.5, 3.5)
    return float(low + value * (high - low))


@dataclass(frozen=True)
class IndustrialAugmentPreset:
    translate_prob: float
    translate_frac: float
    scale_prob: float
    scale_min: float
    scale_max: float
    shear_prob: float
    shear_deg: float
    rotate_prob: float
    rotate_deg: float
    perspective_prob: float
    perspective_frac: float
    exposure_prob: float
    brightness_min: float
    brightness_max: float
    contrast_min: float
    contrast_max: float
    gamma_min: float
    gamma_max: float
    highlight_prob: float
    highlight_count_min: int
    highlight_count_max: int
    highlight_area_min: float
    highlight_area_max: float
    highlight_gain_min: float
    highlight_gain_max: float
    blur_prob: float
    blur_radius_min: float
    blur_radius_max: float
    motion_blur_prob: float
    motion_kernel_min: int
    motion_kernel_max: int
    color_cast_prob: float
    channel_scale_min: float
    channel_scale_max: float
    saturation_prob: float
    saturation_min: float
    saturation_max: float
    shadow_prob: float
    shadow_count_min: int
    shadow_count_max: int
    shadow_area_min: float
    shadow_area_max: float
    shadow_strength_min: float
    shadow_strength_max: float
    offline_min_effects: int
    offline_max_effects: int


PRESETS: Dict[str, IndustrialAugmentPreset] = {
    "light": IndustrialAugmentPreset(
        translate_prob=0.4,
        translate_frac=0.025,
        scale_prob=0.4,
        scale_min=0.975,
        scale_max=1.025,
        shear_prob=0.2,
        shear_deg=1.5,
        rotate_prob=0.1,
        rotate_deg=1.0,
        perspective_prob=0.15,
        perspective_frac=0.015,
        exposure_prob=0.45,
        brightness_min=0.85,
        brightness_max=1.15,
        contrast_min=0.9,
        contrast_max=1.1,
        gamma_min=0.92,
        gamma_max=1.08,
        highlight_prob=0.2,
        highlight_count_min=1,
        highlight_count_max=2,
        highlight_area_min=0.01,
        highlight_area_max=0.035,
        highlight_gain_min=20.0,
        highlight_gain_max=50.0,
        blur_prob=0.15,
        blur_radius_min=0.6,
        blur_radius_max=1.0,
        motion_blur_prob=0.05,
        motion_kernel_min=3,
        motion_kernel_max=5,
        color_cast_prob=0.25,
        channel_scale_min=0.94,
        channel_scale_max=1.06,
        saturation_prob=0.12,
        saturation_min=0.94,
        saturation_max=1.06,
        shadow_prob=0.2,
        shadow_count_min=1,
        shadow_count_max=1,
        shadow_area_min=0.05,
        shadow_area_max=0.12,
        shadow_strength_min=0.65,
        shadow_strength_max=0.85,
        offline_min_effects=1,
        offline_max_effects=2,
    ),
    "medium": IndustrialAugmentPreset(
        translate_prob=0.5,
        translate_frac=0.04,
        scale_prob=0.5,
        scale_min=0.95,
        scale_max=1.05,
        shear_prob=0.3,
        shear_deg=3.0,
        rotate_prob=0.2,
        rotate_deg=2.0,
        perspective_prob=0.25,
        perspective_frac=0.03,
        exposure_prob=0.6,
        brightness_min=0.75,
        brightness_max=1.25,
        contrast_min=0.85,
        contrast_max=1.2,
        gamma_min=0.85,
        gamma_max=1.15,
        highlight_prob=0.35,
        highlight_count_min=1,
        highlight_count_max=3,
        highlight_area_min=0.01,
        highlight_area_max=0.06,
        highlight_gain_min=20.0,
        highlight_gain_max=70.0,
        blur_prob=0.25,
        blur_radius_min=0.6,
        blur_radius_max=1.5,
        motion_blur_prob=0.1,
        motion_kernel_min=3,
        motion_kernel_max=7,
        color_cast_prob=0.4,
        channel_scale_min=0.9,
        channel_scale_max=1.1,
        saturation_prob=0.2,
        saturation_min=0.9,
        saturation_max=1.1,
        shadow_prob=0.35,
        shadow_count_min=1,
        shadow_count_max=2,
        shadow_area_min=0.05,
        shadow_area_max=0.2,
        shadow_strength_min=0.5,
        shadow_strength_max=0.85,
        offline_min_effects=2,
        offline_max_effects=3,
    ),
    "strong": IndustrialAugmentPreset(
        translate_prob=0.65,
        translate_frac=0.06,
        scale_prob=0.65,
        scale_min=0.92,
        scale_max=1.08,
        shear_prob=0.45,
        shear_deg=4.0,
        rotate_prob=0.25,
        rotate_deg=2.0,
        perspective_prob=0.4,
        perspective_frac=0.05,
        exposure_prob=0.75,
        brightness_min=0.65,
        brightness_max=1.35,
        contrast_min=0.8,
        contrast_max=1.3,
        gamma_min=0.75,
        gamma_max=1.25,
        highlight_prob=0.6,
        highlight_count_min=1,
        highlight_count_max=3,
        highlight_area_min=0.015,
        highlight_area_max=0.09,
        highlight_gain_min=35.0,
        highlight_gain_max=95.0,
        blur_prob=0.35,
        blur_radius_min=0.8,
        blur_radius_max=2.0,
        motion_blur_prob=0.16,
        motion_kernel_min=5,
        motion_kernel_max=9,
        color_cast_prob=0.55,
        channel_scale_min=0.82,
        channel_scale_max=1.18,
        saturation_prob=0.3,
        saturation_min=0.85,
        saturation_max=1.18,
        shadow_prob=0.6,
        shadow_count_min=1,
        shadow_count_max=2,
        shadow_area_min=0.08,
        shadow_area_max=0.25,
        shadow_strength_min=0.35,
        shadow_strength_max=0.8,
        offline_min_effects=2,
        offline_max_effects=4,
    ),
}


class IndustrialAugmentor:
    def __init__(self, strength: str = "medium", offline_mode: bool = False):
        if strength not in PRESETS:
            raise ValueError(f"Unsupported industrial augmentation strength: {strength}")
        self.preset = PRESETS[strength]
        self.offline_mode = offline_mode

    def __call__(self, image: Image.Image, mask: Image.Image, rng: np.random.Generator) -> Tuple[Image.Image, Image.Image]:
        image = image.convert("RGB")
        mask = mask.convert("L")

        image, mask = self._apply_geometric(image, mask, rng)
        image = self._apply_photometric(image, rng)
        return image, mask

    def _apply_geometric(self, image: Image.Image, mask: Image.Image, rng: np.random.Generator):
        width, height = image.size
        translate_x = 0.0
        translate_y = 0.0
        scale = 1.0
        shear_x = 0.0
        rotate_deg = 0.0

        if rng.random() < self.preset.translate_prob:
            translate_x = float(rng.uniform(-self.preset.translate_frac, self.preset.translate_frac) * width)
            translate_y = float(rng.uniform(-self.preset.translate_frac, self.preset.translate_frac) * height)

        if rng.random() < self.preset.scale_prob:
            scale = float(rng.uniform(self.preset.scale_min, self.preset.scale_max))

        if rng.random() < self.preset.shear_prob:
            shear_x = float(np.deg2rad(rng.uniform(-self.preset.shear_deg, self.preset.shear_deg)))

        if rng.random() < self.preset.rotate_prob:
            rotate_deg = float(rng.uniform(-self.preset.rotate_deg, self.preset.rotate_deg))

        if any(abs(value) > 1e-6 for value in (translate_x, translate_y, shear_x, rotate_deg)) or abs(scale - 1.0) > 1e-6:
            affine_coeffs = (
                1.0 / scale,
                float(np.tan(-shear_x)),
                -translate_x,
                0.0,
                1.0 / scale,
                -translate_y,
            )
            image = image.transform(
                image.size,
                Image.Transform.AFFINE,
                affine_coeffs,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            mask = mask.transform(
                mask.size,
                Image.Transform.AFFINE,
                affine_coeffs,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )

            if abs(rotate_deg) > 1e-6:
                image = image.rotate(
                    rotate_deg,
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=(0, 0, 0),
                )
                mask = mask.rotate(
                    rotate_deg,
                    resample=Image.Resampling.NEAREST,
                    fillcolor=0,
                )

        if rng.random() < self.preset.perspective_prob:
            max_dx = self.preset.perspective_frac * width
            max_dy = self.preset.perspective_frac * height
            dst_points = [
                (rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
                (width - 1 + rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
                (width - 1 + rng.uniform(-max_dx, max_dx), height - 1 + rng.uniform(-max_dy, max_dy)),
                (rng.uniform(-max_dx, max_dx), height - 1 + rng.uniform(-max_dy, max_dy)),
            ]
            src_points = [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]
            coeffs = _find_perspective_coeffs(src_points, dst_points)
            image = image.transform(
                image.size,
                Image.Transform.PERSPECTIVE,
                coeffs,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            mask = mask.transform(
                mask.size,
                Image.Transform.PERSPECTIVE,
                coeffs,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )

        return image, mask

    def _apply_photometric(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        operations = []

        if rng.random() < self.preset.exposure_prob:
            operations.append("exposure")
        if rng.random() < self.preset.color_cast_prob:
            operations.append("color_cast")
        if rng.random() < self.preset.saturation_prob:
            operations.append("saturation")
        if rng.random() < self.preset.highlight_prob:
            operations.append("highlights")
        if rng.random() < self.preset.shadow_prob:
            operations.append("shadows")
        if rng.random() < self.preset.blur_prob:
            operations.append("gaussian_blur")
        if rng.random() < self.preset.motion_blur_prob:
            operations.append("motion_blur")

        if self.offline_mode:
            guaranteed = int(rng.integers(self.preset.offline_min_effects, self.preset.offline_max_effects + 1))
            required_pool = [
                "exposure",
                "color_cast",
                "saturation",
                "highlights",
                "shadows",
                "gaussian_blur",
                "motion_blur",
            ]
            for effect in rng.permutation(required_pool).tolist():
                if len(set(operations)) >= guaranteed:
                    break
                operations.append(effect)

        for operation in dict.fromkeys(operations):
            if operation == "exposure":
                brightness = float(rng.uniform(self.preset.brightness_min, self.preset.brightness_max))
                contrast = float(rng.uniform(self.preset.contrast_min, self.preset.contrast_max))
                gamma = float(rng.uniform(self.preset.gamma_min, self.preset.gamma_max))
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
                image = self._apply_gamma(image, gamma)
            elif operation == "color_cast":
                image = self._apply_color_cast(image, rng)
            elif operation == "saturation":
                saturation = float(rng.uniform(self.preset.saturation_min, self.preset.saturation_max))
                image = ImageEnhance.Color(image).enhance(saturation)
            elif operation == "highlights":
                image = self._apply_highlights(image, rng)
            elif operation == "shadows":
                image = self._apply_shadows(image, rng)
            elif operation == "gaussian_blur":
                radius = float(rng.uniform(self.preset.blur_radius_min, self.preset.blur_radius_max))
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            elif operation == "motion_blur":
                image = self._apply_motion_blur(image, rng)

        return image

    @staticmethod
    def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.power(np.clip(image_array, 0.0, 1.0), gamma)
        return Image.fromarray(_clip_uint8(image_array * 255.0), mode="RGB")

    def _apply_color_cast(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32)
        channel_scale = rng.uniform(self.preset.channel_scale_min, self.preset.channel_scale_max, size=3)
        image_array = image_array * channel_scale.reshape(1, 1, 3)
        return Image.fromarray(_clip_uint8(image_array), mode="RGB")

    def _apply_highlights(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        width, height = image.size
        base = np.asarray(image, dtype=np.float32)
        result = base.copy()

        count = int(rng.integers(self.preset.highlight_count_min, self.preset.highlight_count_max + 1))
        for _ in range(count):
            area_ratio = float(rng.uniform(self.preset.highlight_area_min, self.preset.highlight_area_max))
            target_area = area_ratio * width * height
            major = max(8.0, np.sqrt(target_area) * float(rng.uniform(0.8, 1.6)))
            minor = max(6.0, target_area / major)
            if rng.random() < 0.5:
                major *= float(rng.uniform(1.1, 1.5))
                minor *= float(rng.uniform(0.6, 0.95))

            cx = _sample_center_bias(rng, width)
            cy = _sample_center_bias(rng, height)
            angle = float(rng.uniform(-30.0, 30.0))

            def draw_fn(draw):
                draw.ellipse(
                    [cx - major / 2, cy - minor / 2, cx + major / 2, cy + minor / 2],
                    fill=255,
                )

            alpha_mask = _soft_mask_from_draw(draw_fn, (width, height), blur_radius=max(2.0, minor * 0.12))
            alpha_mask = np.asarray(
                Image.fromarray(_clip_uint8(alpha_mask * 255.0), mode="L").rotate(angle, resample=Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0

            gain = float(rng.uniform(self.preset.highlight_gain_min, self.preset.highlight_gain_max))
            target = np.clip(result + gain, 0.0, 255.0)
            result = _blend_local_effect(result, target, alpha_mask)

        return Image.fromarray(result, mode="RGB")

    def _apply_shadows(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        width, height = image.size
        base = np.asarray(image, dtype=np.float32)
        result = base.copy()
        count = int(rng.integers(self.preset.shadow_count_min, self.preset.shadow_count_max + 1))

        for _ in range(count):
            area_ratio = float(rng.uniform(self.preset.shadow_area_min, self.preset.shadow_area_max))
            target_area = area_ratio * width * height
            major = max(16.0, np.sqrt(target_area) * float(rng.uniform(1.0, 1.8)))
            minor = max(12.0, target_area / major)
            angle = float(rng.uniform(-45.0, 45.0))
            cx = _sample_center_bias(rng, width)
            cy = _sample_center_bias(rng, height)

            def draw_fn(draw):
                draw.rounded_rectangle(
                    [cx - major / 2, cy - minor / 2, cx + major / 2, cy + minor / 2],
                    radius=max(8.0, min(major, minor) * 0.15),
                    fill=255,
                )

            alpha_mask = _soft_mask_from_draw(draw_fn, (width, height), blur_radius=max(6.0, minor * 0.2))
            alpha_mask = np.asarray(
                Image.fromarray(_clip_uint8(alpha_mask * 255.0), mode="L").rotate(angle, resample=Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0

            darkness = float(rng.uniform(self.preset.shadow_strength_min, self.preset.shadow_strength_max))
            target = result * darkness
            result = _blend_local_effect(result, target, alpha_mask)

        return Image.fromarray(result, mode="RGB")

    def _apply_motion_blur(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        if cv2 is None:
            return image

        image_array = np.asarray(image, dtype=np.float32)
        kernel_size = int(rng.integers(self.preset.motion_kernel_min, self.preset.motion_kernel_max + 1))
        kernel_size = max(3, kernel_size | 1)
        base = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        base[kernel_size // 2, :] = 1.0 / kernel_size
        angle = float(rng.uniform(-20.0, 20.0))
        kernel_image = Image.fromarray(_clip_uint8(base * 255.0), mode="L").rotate(
            angle, resample=Image.Resampling.BILINEAR
        )
        kernel = np.asarray(kernel_image, dtype=np.float32)
        kernel_sum = float(kernel.sum())
        if kernel_sum <= 1e-6:
            return image
        kernel /= kernel_sum
        result = cv2.filter2D(image_array, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        return Image.fromarray(_clip_uint8(result), mode="RGB")


def build_train_augmentor(train_augment: str, augment_strength: str, offline_mode: bool = False):
    if train_augment == "none":
        return None
    if train_augment != "industrial":
        raise ValueError(f"Unsupported train augmentation mode: {train_augment}")
    return IndustrialAugmentor(strength=augment_strength, offline_mode=offline_mode)
