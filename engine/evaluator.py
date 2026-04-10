"""Evaluator for WireCR-HQInstSAM validation and standalone evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from utils.metrics_v2 import apply_classwise_mask_nms, search_thresholds

__all__ = ["WireCRHQInstSAMEvaluator"]


class WireCRHQInstSAMEvaluator:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
        score_grid_label: Sequence[float] = (0.2, 0.35, 0.5, 0.65),
        score_grid_hole: Sequence[float] = (0.2, 0.35, 0.5, 0.65),
        mask_prob_grid: Sequence[float] = (0.4, 0.5, 0.6),
        nms_grid_label: Sequence[float] = (0.5, 0.6, 0.7),
        nms_grid_hole: Sequence[float] = (0.45, 0.55, 0.65),
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.score_grid_label = tuple(float(value) for value in score_grid_label)
        self.score_grid_hole = tuple(float(value) for value in score_grid_hole)
        self.mask_prob_grid = tuple(float(value) for value in mask_prob_grid)
        self.nms_grid_label = tuple(float(value) for value in nms_grid_label)
        self.nms_grid_hole = tuple(float(value) for value in nms_grid_hole)

    @staticmethod
    def _move_targets(batch_targets: Sequence[Mapping[str, Any]], device: torch.device) -> list[dict[str, Any]]:
        moved = []
        for target in batch_targets:
            item = {}
            for key, value in target.items():
                item[key] = value.to(device) if torch.is_tensor(value) else value
            moved.append(item)
        return moved

    @staticmethod
    def _detach_fused_batch(fused_batch: Mapping[str, Any]) -> dict[str, Any]:
        detached = {}
        for key, value in fused_batch.items():
            if torch.is_tensor(value):
                detached[key] = value.detach().cpu()
            elif isinstance(value, list):
                detached[key] = copy.deepcopy(value)
            else:
                detached[key] = value
        return detached

    @staticmethod
    def _target_to_cpu(target: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in target.items():
            result[key] = value.detach().cpu() if torch.is_tensor(value) else value
        return result

    @staticmethod
    def _resolve_source_size(record: Mapping[str, Any]) -> tuple[int, int]:
        crop_box = record.get("crop_box")
        if crop_box is None:
            orig_height, orig_width = (int(value) for value in record["orig_size"])
            return orig_height, orig_width
        x1, y1, x2, y2 = [int(round(float(value))) for value in crop_box]
        return max(y2 - y1, 1), max(x2 - x1, 1)

    @staticmethod
    def _project_logits_to_original_space(
        mask_logits: torch.Tensor,
        *,
        processed_size: tuple[int, int],
        orig_size: tuple[int, int],
        crop_box: Sequence[float] | None,
    ) -> torch.Tensor:
        processed_height, processed_width = (int(value) for value in processed_size)
        orig_height, orig_width = (int(value) for value in orig_size)
        if crop_box is None:
            crop_x1 = 0
            crop_y1 = 0
            source_height = orig_height
            source_width = orig_width
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = [int(round(float(value))) for value in crop_box]
            source_height = max(crop_y2 - crop_y1, 1)
            source_width = max(crop_x2 - crop_x1, 1)

        scale = min(processed_width / float(source_width), processed_height / float(source_height))
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        pad_left = max((processed_width - resized_width) // 2, 0)
        pad_top = max((processed_height - resized_height) // 2, 0)

        cropped = mask_logits[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ]
        restored = F.interpolate(
            cropped.unsqueeze(0).unsqueeze(0).float(),
            size=(source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        full_logits = restored.new_full((orig_height, orig_width), fill_value=-20.0)
        full_logits[crop_y1 : crop_y1 + source_height, crop_x1 : crop_x1 + source_width] = restored
        return full_logits

    @staticmethod
    def _box_from_mask(mask: torch.Tensor) -> torch.Tensor:
        ys, xs = torch.where(mask)
        if xs.numel() == 0 or ys.numel() == 0:
            return torch.zeros((4,), dtype=torch.float32)
        return torch.tensor(
            [
                float(xs.min().item()),
                float(ys.min().item()),
                float(xs.max().item() + 1),
                float(ys.max().item() + 1),
            ],
            dtype=torch.float32,
        )

    def collect_records(self, data_loader) -> list[dict[str, Any]]:
        self.model.eval()
        records = []
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                targets = self._move_targets(batch["instances"], self.device)
                outputs = self.model(
                    images,
                    targets=targets,
                    processed_sizes=batch["processed_size"],
                    prompt_source="pred",
                )
                for batch_index, fused_batch in enumerate(outputs["eval_dict"]["fused_batches"]):
                    records.append(
                        {
                            "image_id": int(batch["image_id"][batch_index]),
                            "image_path": batch["image_path"][batch_index],
                            "orig_size": tuple(batch["orig_size"][batch_index]),
                            "processed_size": tuple(batch["processed_size"][batch_index]),
                            "crop_box": batch["crop_box"][batch_index],
                            "fused_batch": self._detach_fused_batch(fused_batch),
                            "target": self._target_to_cpu(targets[batch_index]),
                        }
                    )
        return records

    @classmethod
    def _select_instances_from_record(
        cls,
        record: Mapping[str, Any],
        thresholds: Mapping[str, float],
        *,
        output_space: str,
    ) -> list[dict[str, Any]]:
        fused_batch = record["fused_batch"]
        if int(fused_batch["labels"].numel()) == 0:
            return []

        refined_logits = fused_batch["refined_mask_logits"][:, 0]
        processed_size = tuple(int(value) for value in record["processed_size"])
        upsampled_logits = F.interpolate(
            refined_logits.unsqueeze(1).float(),
            size=processed_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        candidates = []
        for index in range(int(fused_batch["labels"].shape[0])):
            label = int(fused_batch["labels"][index].item())
            score = float(fused_batch["instance_scores"][index].item())
            score_threshold = thresholds["score_thresh_label"] if label == 1 else thresholds["score_thresh_hole"]
            if score < score_threshold:
                continue
            if output_space == "processed":
                mask_logits = upsampled_logits[index]
            elif output_space == "orig":
                mask_logits = cls._project_logits_to_original_space(
                    upsampled_logits[index],
                    processed_size=processed_size,
                    orig_size=tuple(int(value) for value in record["orig_size"]),
                    crop_box=record.get("crop_box"),
                )
            else:
                raise ValueError(f"Unsupported output_space: {output_space}")
            binary_mask = mask_logits.sigmoid() > float(thresholds.get("mask_prob_thresh", 0.5))
            if int(binary_mask.sum().item()) <= 0:
                continue
            box = cls._box_from_mask(binary_mask)
            candidates.append(
                {
                    "label": label,
                    "score": score,
                    "mask": binary_mask,
                    "mask_logits": mask_logits,
                    "box": box,
                }
            )
        return apply_classwise_mask_nms(
            candidates,
            iou_threshold_label=thresholds["mask_nms_iou_label"],
            iou_threshold_hole=thresholds["mask_nms_iou_hole"],
        )

    @classmethod
    def _select_processed_instances_from_record(
        cls,
        record: Mapping[str, Any],
        thresholds: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        return cls._select_instances_from_record(record, thresholds, output_space="processed")

    @classmethod
    def _select_coco_instances_from_record(
        cls,
        record: Mapping[str, Any],
        thresholds: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        return cls._select_instances_from_record(record, thresholds, output_space="orig")

    def evaluate_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        coco_gt = None
        return search_thresholds(
            records,
            coco_gt=coco_gt,
            selector=self._select_processed_instances_from_record,
            coco_selector=self._select_coco_instances_from_record,
            score_grid_label=self.score_grid_label,
            score_grid_hole=self.score_grid_hole,
            mask_prob_grid=self.mask_prob_grid,
            nms_grid_label=self.nms_grid_label,
            nms_grid_hole=self.nms_grid_hole,
        )

    def evaluate(self, data_loader, *, output_path: str | Path | None = None) -> dict[str, Any]:
        records = self.collect_records(data_loader)
        if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "coco"):
            result = search_thresholds(
                records,
                coco_gt=data_loader.dataset.coco,
                selector=self._select_processed_instances_from_record,
                coco_selector=self._select_coco_instances_from_record,
                score_grid_label=self.score_grid_label,
                score_grid_hole=self.score_grid_hole,
                mask_prob_grid=self.mask_prob_grid,
                nms_grid_label=self.nms_grid_label,
                nms_grid_hole=self.nms_grid_hole,
            )
        else:
            result = self.evaluate_records(records)

        if output_path is not None:
            output_file = Path(output_path).expanduser().resolve()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        return result
