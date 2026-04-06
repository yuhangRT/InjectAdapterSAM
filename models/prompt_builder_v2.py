"""Prompt builder v2 for HQ refinement prompts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.matcher import box_iou

__all__ = ["PromptBuilderV2"]


class PromptBuilderV2(nn.Module):
    """Build class-aware prompts from gt/pred/mixed coarse instance inputs."""

    def __init__(
        self,
        *,
        dense_prompt_downscale: int = 4,
        gt_box_jitter: float = 0.05,
        joint_gt_ratio_start: float = 0.7,
        joint_gt_ratio_end: float = 0.1,
    ) -> None:
        super().__init__()
        self.dense_prompt_downscale = int(dense_prompt_downscale)
        self.gt_box_jitter = float(gt_box_jitter)
        self.joint_gt_ratio_start = float(joint_gt_ratio_start)
        self.joint_gt_ratio_end = float(joint_gt_ratio_end)

    def resolve_joint_gt_ratio(self, progress: float | None) -> float:
        if progress is None:
            return self.joint_gt_ratio_start
        progress = float(max(0.0, min(1.0, progress)))
        return self.joint_gt_ratio_start + (self.joint_gt_ratio_end - self.joint_gt_ratio_start) * progress

    @staticmethod
    def _to_float_tensor(value: torch.Tensor | Sequence[float] | Sequence[Sequence[float]], device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    @staticmethod
    def _to_long_tensor(value: torch.Tensor | Sequence[int], device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=device, dtype=torch.long)
        return torch.as_tensor(value, dtype=torch.long, device=device)

    @staticmethod
    def _boxes_to_absolute_xyxy(boxes: torch.Tensor, processed_size: tuple[int, int]) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(-1, 4).float()

        height, width = processed_size
        boxes = boxes.float()
        if float(boxes.detach().amax().item()) <= 1.0:
            scale = boxes.new_tensor([width, height, width, height])
            boxes = boxes * scale

        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        boxes = torch.stack(
            (
                torch.minimum(x0, x1).clamp(0.0, float(width - 1)),
                torch.minimum(y0, y1).clamp(0.0, float(height - 1)),
                torch.maximum(x0, x1).clamp(0.0, float(width - 1)),
                torch.maximum(y0, y1).clamp(0.0, float(height - 1)),
            ),
            dim=-1,
        )
        return boxes

    def _dense_prompt_size(
        self,
        processed_size: tuple[int, int],
        source_masks: torch.Tensor | None,
    ) -> tuple[int, int]:
        return (
            max(int(processed_size[0] // self.dense_prompt_downscale), 4),
            max(int(processed_size[1] // self.dense_prompt_downscale), 4),
        )

    def _to_dense_prompt_logits(
        self,
        source_masks: torch.Tensor,
        *,
        processed_size: tuple[int, int],
    ) -> torch.Tensor:
        if source_masks.dim() == 3:
            source_masks = source_masks.unsqueeze(1)
        if source_masks.dim() != 4:
            raise ValueError(f"Expected source masks in NCHW or NHW format, got {tuple(source_masks.shape)}.")

        dense_size = self._dense_prompt_size(processed_size, source_masks[:, 0])
        dense = source_masks.float()
        if tuple(dense.shape[-2:]) != dense_size:
            dense = F.interpolate(dense, size=dense_size, mode="bilinear", align_corners=False)

        min_value = float(dense.detach().amin().item()) if dense.numel() else 0.0
        max_value = float(dense.detach().amax().item()) if dense.numel() else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            dense = dense.clamp(1e-4, 1.0 - 1e-4)
            dense = torch.logit(dense)
        return dense

    @staticmethod
    def _normalize_axis(principal_axis: torch.Tensor | None) -> torch.Tensor | None:
        if principal_axis is None or principal_axis.numel() != 2:
            return None
        axis = principal_axis.float()
        norm = torch.linalg.norm(axis)
        if float(norm.item()) <= 1e-6:
            return None
        return axis / norm

    @staticmethod
    def _axis_from_oriented_box(oriented_box: torch.Tensor | None) -> torch.Tensor | None:
        if oriented_box is None or oriented_box.numel() != 5:
            return None
        angle_deg = float(oriented_box[-1].item())
        angle = torch.tensor(angle_deg * torch.pi / 180.0, dtype=torch.float32, device=oriented_box.device)
        return torch.stack((torch.cos(angle), torch.sin(angle)))

    @staticmethod
    def _center_from_box(box: torch.Tensor) -> torch.Tensor:
        return torch.stack(((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5))

    @staticmethod
    def _clip_points(points: torch.Tensor, processed_size: tuple[int, int]) -> torch.Tensor:
        height, width = processed_size
        points = points.float()
        x = points[..., 0].clamp(0.0, float(width - 1))
        y = points[..., 1].clamp(0.0, float(height - 1))
        return torch.stack((x, y), dim=-1)

    def _jitter_box(
        self,
        box: torch.Tensor,
        *,
        processed_size: tuple[int, int],
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        center = self._center_from_box(box)
        width = max(float((box[2] - box[0]).item()), 1.0)
        height = max(float((box[3] - box[1]).item()), 1.0)
        jitter = torch.empty((4,), dtype=torch.float32, device=box.device)
        jitter.uniform_(-self.gt_box_jitter, self.gt_box_jitter, generator=generator)
        jitter = jitter * box.new_tensor([width, height, width, height])
        jittered = box + jitter
        return self._boxes_to_absolute_xyxy(jittered.unsqueeze(0), processed_size)[0]

    def _resolve_axis(
        self,
        *,
        box: torch.Tensor,
        principal_axis: torch.Tensor | None,
        oriented_box: torch.Tensor | None,
    ) -> tuple[torch.Tensor, bool, bool]:
        axis = self._normalize_axis(principal_axis)
        if axis is not None:
            return axis.to(box.device), True, False
        axis = self._axis_from_oriented_box(oriented_box)
        if axis is not None:
            axis = self._normalize_axis(axis)
            if axis is not None:
                return axis.to(box.device), False, True

        width = float((box[2] - box[0]).item())
        height = float((box[3] - box[1]).item())
        if width >= height:
            return box.new_tensor([1.0, 0.0]), False, False
        return box.new_tensor([0.0, 1.0]), False, False

    def _build_label_points(
        self,
        *,
        box: torch.Tensor,
        processed_size: tuple[int, int],
        principal_axis: torch.Tensor | None,
        oriented_box: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        center = self._center_from_box(box)
        axis, used_principal_axis, used_oriented_box = self._resolve_axis(
            box=box,
            principal_axis=principal_axis,
            oriented_box=oriented_box,
        )
        perpendicular = box.new_tensor([-axis[1], axis[0]])
        half_extent = max(float(min(box[2] - box[0], box[3] - box[1]).item()) * 0.25, 1.0)
        negative_extent = max(float(max(box[2] - box[0], box[3] - box[1]).item()) * 0.55, 1.0)

        positive_points = torch.stack(
            (
                center - axis * half_extent,
                center,
                center + axis * half_extent,
            ),
            dim=0,
        )
        negative_points = torch.stack(
            (
                center - perpendicular * negative_extent,
                center + perpendicular * negative_extent,
            ),
            dim=0,
        )
        points = torch.cat((positive_points, negative_points), dim=0)
        point_labels = box.new_tensor([1, 1, 1, 0, 0], dtype=torch.long)
        meta = {
            "sampling_strategy": "label_axis_3pos_2neg",
            "num_positive_points": 3,
            "num_negative_points": 2,
            "used_principal_axis_aux": used_principal_axis,
            "used_oriented_box_aux": used_oriented_box,
        }
        return self._clip_points(points, processed_size), point_labels, meta

    def _build_hole_points(
        self,
        *,
        box: torch.Tensor,
        processed_size: tuple[int, int],
        oriented_box: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if oriented_box is not None and oriented_box.numel() == 5:
            center = oriented_box[:2].to(box.device, dtype=torch.float32)
            used_oriented_box = True
        else:
            center = self._center_from_box(box)
            used_oriented_box = False

        half_width = max(float((box[2] - box[0]).item()) * 0.55, 1.0)
        half_height = max(float((box[3] - box[1]).item()) * 0.55, 1.0)
        positive = center.unsqueeze(0)
        negatives = torch.stack(
            (
                center + box.new_tensor([0.0, -half_height]),
                center + box.new_tensor([half_width, 0.0]),
                center + box.new_tensor([0.0, half_height]),
                center + box.new_tensor([-half_width, 0.0]),
            ),
            dim=0,
        )
        points = torch.cat((positive, negatives), dim=0)
        point_labels = box.new_tensor([1, 0, 0, 0, 0], dtype=torch.long)
        meta = {
            "sampling_strategy": "hole_center_1pos_4neg",
            "num_positive_points": 1,
            "num_negative_points": 4,
            "used_principal_axis_aux": False,
            "used_oriented_box_aux": used_oriented_box,
        }
        return self._clip_points(points, processed_size), point_labels, meta

    def _normalize_instance_inputs(
        self,
        instances: Mapping[str, Any] | None,
        *,
        processed_size: tuple[int, int],
        device: torch.device,
    ) -> Dict[str, Any]:
        if instances is None:
            empty = torch.zeros((0,), device=device)
            return {
                "boxes": empty.reshape(0, 4),
                "mask_logits": empty.reshape(0, 1, 0, 0),
                "labels": empty.to(dtype=torch.long),
                "oriented_boxes": empty.reshape(0, 5),
                "principal_axes": empty.reshape(0, 2),
                "processed_size": processed_size,
            }

        source_processed_size = tuple(int(value) for value in instances.get("processed_size", processed_size))
        boxes = self._boxes_to_absolute_xyxy(
            self._to_float_tensor(instances.get("boxes", torch.zeros((0, 4))), device),
            source_processed_size,
        )
        labels = self._to_long_tensor(instances.get("labels", torch.zeros((boxes.shape[0],), dtype=torch.long)), device)
        masks = instances.get("mask_logits")
        if masks is None:
            masks = instances.get("masks", torch.zeros((boxes.shape[0], *source_processed_size), dtype=torch.float32))
        mask_logits = self._to_dense_prompt_logits(
            self._to_float_tensor(masks, device),
            processed_size=source_processed_size,
        )

        oriented_boxes_value = instances.get("oriented_boxes")
        if oriented_boxes_value is None:
            oriented_boxes = boxes.new_zeros((boxes.shape[0], 5))
        else:
            oriented_boxes = self._to_float_tensor(oriented_boxes_value, device).reshape(-1, 5)

        principal_axes_value = instances.get("principal_axes")
        if principal_axes_value is None:
            principal_axes = boxes.new_zeros((boxes.shape[0], 2))
        else:
            principal_axes = self._to_float_tensor(principal_axes_value, device).reshape(-1, 2)

        return {
            "boxes": boxes,
            "mask_logits": mask_logits,
            "labels": labels,
            "oriented_boxes": oriented_boxes,
            "principal_axes": principal_axes,
            "processed_size": source_processed_size,
        }

    def _select_source_index(
        self,
        *,
        index: int,
        pred_count: int,
        gt_count: int,
        prompt_source: str,
        gt_ratio: float,
        generator: torch.Generator | None,
    ) -> tuple[str, int] | None:
        if prompt_source == "gt":
            if index >= gt_count:
                return None
            return "gt", index
        if prompt_source == "pred":
            if index >= pred_count:
                return None
            return "pred", index

        use_gt = bool(torch.rand((), generator=generator).item() < gt_ratio)
        if use_gt and index < gt_count:
            return "gt", index
        if index < pred_count:
            return "pred", index
        if index < gt_count:
            return "gt", index
        return None

    @staticmethod
    def _mask_iou_from_logits(pred_logits: torch.Tensor, gt_logits: torch.Tensor) -> float:
        pred_mask = pred_logits > 0
        gt_mask = gt_logits > 0
        intersection = float((pred_mask & gt_mask).float().sum().item())
        union = float((pred_mask | gt_mask).float().sum().item())
        if union <= 0.0:
            return 0.0
        return intersection / union

    def _match_pred_to_gt(
        self,
        *,
        pred: Mapping[str, Any],
        gt: Mapping[str, Any],
        pred_index: int,
    ) -> tuple[int, float, float]:
        gt_labels = gt["labels"]
        if gt_labels.numel() == 0:
            return -1, 0.0, 0.0

        pred_label = int(pred["labels"][pred_index].item()) if pred["labels"].numel() > pred_index else 0
        same_label = torch.nonzero(gt_labels == pred_label, as_tuple=False).flatten()
        if same_label.numel() == 0:
            return -1, 0.0, 0.0

        pred_box = pred["boxes"][pred_index : pred_index + 1]
        gt_boxes = gt["boxes"][same_label]
        box_overlaps = box_iou(pred_box, gt_boxes)[0]

        pred_dense = pred["mask_logits"][pred_index, 0]
        gt_dense = gt["mask_logits"][same_label, 0]
        mask_overlaps = pred_dense.new_zeros((same_label.numel(),), dtype=torch.float32)
        for idx in range(int(same_label.numel())):
            mask_overlaps[idx] = self._mask_iou_from_logits(pred_dense, gt_dense[idx])

        combined = box_overlaps + 0.5 * mask_overlaps
        best_local = int(combined.argmax().item())
        best_gt_index = int(same_label[best_local].item())
        best_box_iou = float(box_overlaps[best_local].item())
        best_mask_iou = float(mask_overlaps[best_local].item())
        if max(best_box_iou, best_mask_iou) < 0.05:
            return -1, best_box_iou, best_mask_iou
        return best_gt_index, best_box_iou, best_mask_iou

    def build_prompts(
        self,
        *,
        pred_instances: Mapping[str, Any] | None = None,
        gt_instances: Mapping[str, Any] | None = None,
        processed_size: tuple[int, int],
        prompt_source: str = "pred",
        joint_progress: float | None = None,
        gt_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Dict[str, Any]:
        prompt_source = str(prompt_source).strip().lower()
        if prompt_source not in {"gt", "pred", "mixed"}:
            raise ValueError(f"Unsupported prompt_source: {prompt_source}")

        device = torch.device("cpu")
        for instances in (pred_instances, gt_instances):
            if instances is not None:
                candidate = instances.get("boxes")
                if torch.is_tensor(candidate):
                    device = candidate.device
                    break

        pred = self._normalize_instance_inputs(pred_instances, processed_size=processed_size, device=device)
        gt = self._normalize_instance_inputs(gt_instances, processed_size=processed_size, device=device)

        pred_count = int(pred["boxes"].shape[0])
        gt_count = int(gt["boxes"].shape[0])
        if prompt_source == "pred" and pred_count == 0:
            count = 0
        elif prompt_source == "gt" and gt_count == 0:
            count = 0
        else:
            count = max(pred_count, gt_count) if prompt_source == "mixed" else (gt_count if prompt_source == "gt" else pred_count)

        resolved_gt_ratio = float(gt_ratio if gt_ratio is not None else self.resolve_joint_gt_ratio(joint_progress))

        prompt_boxes = []
        dense_prompts = []
        point_coords = []
        point_labels = []
        prompt_meta = []

        for index in range(count):
            selection = self._select_source_index(
                index=index,
                pred_count=pred_count,
                gt_count=gt_count,
                prompt_source=prompt_source,
                gt_ratio=resolved_gt_ratio,
                generator=generator,
            )
            if selection is None:
                continue

            source_kind, source_index = selection
            source = gt if source_kind == "gt" else pred
            box = source["boxes"][source_index]
            dense_prompt = source["mask_logits"][source_index]
            label = int(source["labels"][source_index].item()) if source["labels"].numel() > 0 else 0
            oriented_box = source["oriented_boxes"][source_index] if source["oriented_boxes"].shape[0] > source_index else None
            principal_axis = source["principal_axes"][source_index] if source["principal_axes"].shape[0] > source_index else None
            source_processed_size = source["processed_size"]
            if source_kind == "gt":
                matched_gt_index = int(source_index)
                matched_gt_box_iou = 1.0
                matched_gt_mask_iou = 1.0
            else:
                matched_gt_index, matched_gt_box_iou, matched_gt_mask_iou = self._match_pred_to_gt(
                    pred=pred,
                    gt=gt,
                    pred_index=source_index,
                )

            box_jitter_applied = source_kind == "gt"
            prompt_box = self._jitter_box(box, processed_size=source_processed_size, generator=generator) if box_jitter_applied else box

            if label == 1:
                sampled_points, sampled_labels, meta = self._build_label_points(
                    box=prompt_box,
                    processed_size=source_processed_size,
                    principal_axis=principal_axis,
                    oriented_box=oriented_box,
                )
                class_name = "label_sleeve"
            elif label == 2:
                sampled_points, sampled_labels, meta = self._build_hole_points(
                    box=prompt_box,
                    processed_size=source_processed_size,
                    oriented_box=oriented_box,
                )
                class_name = "empty_terminal"
            else:
                sampled_points = prompt_box.new_zeros((0, 2))
                sampled_labels = prompt_box.new_zeros((0,), dtype=torch.long)
                meta = {
                    "sampling_strategy": "background_skip",
                    "num_positive_points": 0,
                    "num_negative_points": 0,
                    "used_principal_axis_aux": False,
                    "used_oriented_box_aux": False,
                }
                class_name = "background"

            prompt_boxes.append(prompt_box)
            dense_prompts.append(dense_prompt)
            point_coords.append(sampled_points)
            point_labels.append(sampled_labels)
            prompt_meta.append(
                {
                    "prompt_source": prompt_source,
                    "instance_source": source_kind,
                    "source_index": int(source_index),
                    "matched_gt_index": int(matched_gt_index),
                    "matched_gt_box_iou": float(matched_gt_box_iou),
                    "matched_gt_mask_iou": float(matched_gt_mask_iou),
                    "label": label,
                    "class_name": class_name,
                    "dense_prompt_type": "logits",
                    "box_jitter_applied": box_jitter_applied,
                    "processed_size": source_processed_size,
                    **meta,
                    "num_points": int(sampled_labels.numel()),
                }
            )

        if not prompt_boxes:
            dense_height = max(int(processed_size[0] // self.dense_prompt_downscale), 4)
            dense_width = max(int(processed_size[1] // self.dense_prompt_downscale), 4)
            return {
                "boxes_xyxy": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "dense_mask_prompt_logits": torch.zeros((0, 1, dense_height, dense_width), dtype=torch.float32, device=device),
                "point_coords": torch.zeros((0, 0, 2), dtype=torch.float32, device=device),
                "point_labels": torch.zeros((0, 0), dtype=torch.long, device=device),
                "prompt_meta": [],
            }

        return {
            "boxes_xyxy": torch.stack(prompt_boxes, dim=0),
            "dense_mask_prompt_logits": torch.stack(dense_prompts, dim=0),
            "point_coords": torch.stack(point_coords, dim=0),
            "point_labels": torch.stack(point_labels, dim=0),
            "prompt_meta": prompt_meta,
        }
