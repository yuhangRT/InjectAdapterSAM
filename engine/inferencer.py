"""Sliding-window inferencer and exporters for WireCR-HQInstSAM."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageColor, ImageDraw
import torch
import torch.nn.functional as F

from dataset.geometry_utils import oriented_box_from_mask
from utils.metrics_v2 import apply_classwise_mask_nms, mask_iou

__all__ = ["WireCRHQInstSAMInferencer"]


class WireCRHQInstSAMInferencer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
        sliding_window: int = 1024,
        overlap: float = 0.2,
        score_thresh_label: float = 0.5,
        score_thresh_hole: float = 0.5,
        mask_prob_thresh: float = 0.5,
        mask_nms_iou_label: float = 0.6,
        mask_nms_iou_hole: float = 0.5,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.sliding_window = int(sliding_window)
        self.overlap = float(overlap)
        self.score_thresh_label = float(score_thresh_label)
        self.score_thresh_hole = float(score_thresh_hole)
        self.mask_prob_thresh = float(mask_prob_thresh)
        self.mask_nms_iou_label = float(mask_nms_iou_label)
        self.mask_nms_iou_hole = float(mask_nms_iou_hole)

    @staticmethod
    def image_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _window_boxes(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        window = min(self.sliding_window, max(height, width))
        stride = max(int(round(window * (1.0 - self.overlap))), 1)
        y_starts = list(range(0, max(height - window, 0) + 1, stride))
        x_starts = list(range(0, max(width - window, 0) + 1, stride))
        if not y_starts:
            y_starts = [0]
        if not x_starts:
            x_starts = [0]
        if y_starts[-1] != max(height - window, 0):
            y_starts.append(max(height - window, 0))
        if x_starts[-1] != max(width - window, 0):
            x_starts.append(max(width - window, 0))
        return [(x1, y1, min(x1 + window, width), min(y1 + window, height)) for y1 in y_starts for x1 in x_starts]

    def _run_window(self, crop_tensor: torch.Tensor) -> Mapping[str, Any]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                crop_tensor.unsqueeze(0).to(self.device),
                processed_sizes=[tuple(int(value) for value in crop_tensor.shape[-2:])],
                prompt_source="pred",
            )
        return outputs["eval_dict"]["fused_batches"][0]

    def _fused_batch_to_window_instances(
        self,
        fused_batch: Mapping[str, Any],
        *,
        crop_size: tuple[int, int],
        window_box: tuple[int, int, int, int],
        full_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        if int(fused_batch["labels"].numel()) == 0:
            return []

        crop_height, crop_width = crop_size
        full_height, full_width = full_size
        x1, y1, _, _ = window_box
        logits = fused_batch["refined_mask_logits"][:, 0].detach().cpu()
        upsampled_logits = F.interpolate(
            logits.unsqueeze(1).float(),
            size=(crop_height, crop_width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        instances = []
        for index in range(int(fused_batch["labels"].shape[0])):
            label = int(fused_batch["labels"][index].item())
            score = float(fused_batch["instance_scores"][index].item())
            score_threshold = self.score_thresh_label if label == 1 else self.score_thresh_hole
            if score < score_threshold:
                continue
            local_mask = upsampled_logits[index].sigmoid() > self.mask_prob_thresh
            if int(local_mask.sum().item()) <= 0:
                continue
            full_mask = torch.zeros((full_height, full_width), dtype=torch.bool)
            full_mask[y1 : y1 + crop_height, x1 : x1 + crop_width] = local_mask
            local_box = fused_batch["boxes_xyxy"][index].detach().cpu().float()
            full_box = local_box + torch.tensor([x1, y1, x1, y1], dtype=torch.float32)
            instances.append(
                {
                    "label": label,
                    "score": score,
                    "mask": full_mask,
                    "mask_logits": full_mask.float(),
                    "box": full_box,
                    "window_box": window_box,
                }
            )
        return instances

    @staticmethod
    def _center(box: torch.Tensor) -> torch.Tensor:
        return torch.tensor([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=torch.float32)

    def _should_merge_windows(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        if int(a["label"]) != int(b["label"]):
            return False
        iou = mask_iou(a["mask"], b["mask"])
        center_distance = float(torch.linalg.norm(self._center(a["box"]) - self._center(b["box"])).item())
        diag = max(
            float(torch.linalg.norm(a["box"][2:] - a["box"][:2]).item()),
            float(torch.linalg.norm(b["box"][2:] - b["box"][:2]).item()),
            1.0,
        )
        return iou >= 0.3 or (iou >= 0.1 and center_distance <= 0.25 * diag)

    @staticmethod
    def _merge_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        merged_mask = a["mask"] | b["mask"]
        ys, xs = torch.where(merged_mask)
        if xs.numel() == 0 or ys.numel() == 0:
            merged_box = a["box"]
        else:
            merged_box = torch.tensor(
                [float(xs.min().item()), float(ys.min().item()), float(xs.max().item() + 1), float(ys.max().item() + 1)],
                dtype=torch.float32,
            )
        return {
            "label": int(a["label"]),
            "score": float(max(float(a["score"]), float(b["score"]))),
            "mask": merged_mask,
            "mask_logits": merged_mask.float(),
            "box": merged_box,
            "window_box": a.get("window_box"),
        }

    def _cross_window_fuse(self, instances: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted((dict(item) for item in instances), key=lambda item: float(item["score"]), reverse=True)
        fused: list[dict[str, Any]] = []
        while ordered:
            current = ordered.pop(0)
            merged = False
            for index, existing in enumerate(fused):
                if self._should_merge_windows(existing, current):
                    fused[index] = self._merge_pair(existing, current)
                    merged = True
                    break
            if not merged:
                fused.append(current)
        return fused

    def _final_nms(self, instances: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return apply_classwise_mask_nms(
            instances,
            iou_threshold_label=self.mask_nms_iou_label,
            iou_threshold_hole=self.mask_nms_iou_hole,
        )

    def predict_tensor(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        window_instances = []
        for window_box in self._window_boxes(height, width):
            x1, y1, x2, y2 = window_box
            crop_tensor = image_tensor[:, y1:y2, x1:x2]
            fused_batch = self._run_window(crop_tensor)
            window_instances.extend(
                self._fused_batch_to_window_instances(
                    fused_batch,
                    crop_size=(y2 - y1, x2 - x1),
                    window_box=window_box,
                    full_size=(height, width),
                )
            )

        fused_instances = self._cross_window_fuse(window_instances)
        final_instances = self._final_nms(fused_instances)
        return {
            "window_instances": window_instances,
            "fused_instances": fused_instances,
            "instances": final_instances,
        }

    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        return self.predict_tensor(self.image_to_tensor(image))

    @staticmethod
    def _mask_to_png(mask: torch.Tensor, path: Path) -> None:
        image = Image.fromarray((mask.detach().cpu().numpy().astype(np.uint8) * 255), mode="L")
        image.save(path)

    @staticmethod
    def _draw_visualization(image: Image.Image, instances: Sequence[Mapping[str, Any]], path: Path) -> None:
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        color_map = {
            1: ImageColor.getrgb("#ff6b6b"),
            2: ImageColor.getrgb("#1dd1a1"),
        }
        for instance in instances:
            color = color_map.get(int(instance["label"]), (255, 255, 0))
            mask = Image.fromarray(instance["mask"].detach().cpu().numpy().astype(np.uint8) * 96, mode="L")
            solid = Image.new("RGBA", base.size, color + (0,))
            solid.putalpha(mask)
            overlay = Image.alpha_composite(overlay, solid)
        vis = Image.alpha_composite(base, overlay)
        draw = ImageDraw.Draw(vis)
        for instance in instances:
            box = instance["box"].tolist()
            color = color_map.get(int(instance["label"]), (255, 255, 0))
            draw.rectangle(box, outline=color + (255,), width=2)
            draw.text((box[0] + 2, box[1] + 2), f"{instance['label']}:{instance['score']:.2f}", fill=color + (255,))
        vis.convert("RGB").save(path)

    @staticmethod
    def _rectified_crop(image: Image.Image, mask: torch.Tensor) -> Image.Image:
        mask_list = mask.detach().cpu().numpy().astype(np.uint8).tolist()
        center_x, center_y, width, height, angle = oriented_box_from_mask(mask_list)
        if width <= 1.0 or height <= 1.0:
            bbox = mask.nonzero()
            if bbox.numel() == 0:
                return image.copy()
            y_coords = bbox[:, 0]
            x_coords = bbox[:, 1]
            return image.crop((int(x_coords.min()), int(y_coords.min()), int(x_coords.max()) + 1, int(y_coords.max()) + 1))
        rotated = image.rotate(-angle, center=(center_x, center_y), resample=Image.BILINEAR)
        x1 = max(int(round(center_x - width * 0.5)), 0)
        y1 = max(int(round(center_y - height * 0.5)), 0)
        x2 = min(int(round(center_x + width * 0.5)), rotated.width)
        y2 = min(int(round(center_y + height * 0.5)), rotated.height)
        return rotated.crop((x1, y1, x2, y2))

    def export_predictions(
        self,
        *,
        image: Image.Image,
        prediction: Mapping[str, Any],
        output_dir: str | Path,
        stem: str,
    ) -> dict[str, Any]:
        root = Path(output_dir).expanduser().resolve()
        json_dir = root / "json"
        vis_dir = root / "vis"
        mask_dir = root / "masks"
        crop_dir = root / "rectified_crops"
        for directory in (json_dir, vis_dir, mask_dir, crop_dir):
            directory.mkdir(parents=True, exist_ok=True)

        records = []
        for index, instance in enumerate(prediction["instances"]):
            mask_path = mask_dir / f"{stem}_inst_{index:03d}.png"
            self._mask_to_png(instance["mask"], mask_path)
            crop_path = None
            if int(instance["label"]) == 1:
                crop = self._rectified_crop(image, instance["mask"])
                crop_path = crop_dir / f"{stem}_inst_{index:03d}.png"
                crop.save(crop_path)
            records.append(
                {
                    "label": int(instance["label"]),
                    "score": float(instance["score"]),
                    "box": [float(value) for value in instance["box"].tolist()],
                    "mask_path": str(mask_path),
                    "rectified_crop": None if crop_path is None else str(crop_path),
                }
            )

        json_path = json_dir / f"{stem}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump({"instances": records}, handle, ensure_ascii=False, indent=2)
        self._draw_visualization(image, prediction["instances"], vis_dir / f"{stem}.png")
        return {
            "json_path": json_path,
            "records": records,
        }
