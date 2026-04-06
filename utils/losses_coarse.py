"""Coarse DETR-style losses for WireCR-HQInstSAM."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.matcher import HungarianMatcher, generalized_box_iou, normalize_xyxy_boxes, resolve_target_spatial_size

__all__ = ["CoarseLossCriterion"]


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def _balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_pos_weight: float = 256.0,
) -> torch.Tensor:
    positives = targets.sum()
    negatives = targets.numel() - positives
    pos_weight = negatives / positives.clamp(min=1.0)
    pos_weight = pos_weight.clamp(min=1.0, max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=logits.new_tensor(float(pos_weight.item())),
    )


class CoarseLossCriterion(nn.Module):
    """Compute coarse classification, box, mask, and repulsion losses."""

    def __init__(
        self,
        matcher: HungarianMatcher | None = None,
        *,
        num_classes: int = 2,
        eos_coef: float = 0.1,
        loss_weights: Dict[str, float] | None = None,
        aux_loss_weight: float = 1.0,
        label_sleeve_label: int = 1,
    ) -> None:
        super().__init__()
        self.matcher = matcher or HungarianMatcher()
        self.num_classes = num_classes
        self.aux_loss_weight = float(aux_loss_weight)
        self.label_sleeve_label = int(label_sleeve_label)
        self.loss_weights = {
            "cls": 2.0,
            "bbox": 5.0,
            "giou": 2.0,
            "mask_bce": 5.0,
            "mask_dice": 5.0,
            "repulsion": 0.5,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        empty_weight = torch.ones(num_classes + 1, dtype=torch.float32)
        empty_weight[0] = float(eos_coef)
        self.register_buffer("empty_weight", empty_weight, persistent=False)

    @staticmethod
    def _resize_target_masks(target_masks: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        if target_masks.numel() == 0:
            return target_masks.reshape(0, *spatial_size)
        resized = F.interpolate(
            target_masks.unsqueeze(1).float(),
            size=spatial_size,
            mode="nearest",
        )
        return resized[:, 0]

    def _classification_loss(
        self,
        pred_logits: torch.Tensor,
        targets: Sequence[dict[str, torch.Tensor]],
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        target_classes = torch.zeros(
            pred_logits.shape[:2],
            dtype=torch.long,
            device=pred_logits.device,
        )
        for batch_index, (pred_idx, target_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            target_classes[batch_index, pred_idx] = targets[batch_index]["labels"][target_idx].to(pred_logits.device)
        return F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(pred_logits.device),
        )

    def _matched_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        targets: Sequence[dict[str, torch.Tensor]],
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_boxes = []
        target_boxes = []
        pred_masks = []
        target_masks = []
        target_labels = []

        for batch_index, (pred_idx, target_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            pred_boxes.append(outputs["pred_boxes"][batch_index, pred_idx])
            pred_masks.append(outputs["pred_masks"][batch_index, pred_idx])
            current_target = targets[batch_index]
            target_spatial_size = resolve_target_spatial_size(
                current_target,
                fallback=tuple(int(value) for value in outputs["pred_masks"].shape[-2:]),
            )
            resized_masks = self._resize_target_masks(
                current_target["masks"].to(outputs["pred_masks"].device),
                spatial_size=outputs["pred_masks"].shape[-2:],
            )
            target_masks.append(resized_masks[target_idx])
            target_boxes.append(
                normalize_xyxy_boxes(
                    current_target["boxes"].to(outputs["pred_boxes"].device)[target_idx],
                    spatial_size=target_spatial_size,
                )
            )
            target_labels.append(current_target["labels"].to(outputs["pred_boxes"].device)[target_idx])

        if not pred_boxes:
            empty_boxes = outputs["pred_boxes"].new_zeros((0, 4))
            empty_masks = outputs["pred_masks"].new_zeros((0, *outputs["pred_masks"].shape[-2:]))
            empty_labels = outputs["pred_boxes"].new_zeros((0,), dtype=torch.long)
            return empty_boxes, empty_boxes, empty_masks, empty_masks, empty_labels

        return (
            torch.cat(pred_boxes, dim=0),
            torch.cat(target_boxes, dim=0),
            torch.cat(pred_masks, dim=0),
            torch.cat(target_masks, dim=0),
            torch.cat(target_labels, dim=0),
        )

    def _repulsion_loss(self, matched_masks: torch.Tensor, matched_labels: torch.Tensor) -> torch.Tensor:
        label_mask = matched_labels == self.label_sleeve_label
        if int(label_mask.sum().item()) < 2:
            return matched_masks.sum() * 0.0

        selected = matched_masks[label_mask].sigmoid()
        penalties = []
        for idx in range(selected.shape[0]):
            for other_idx in range(idx + 1, selected.shape[0]):
                penalties.append((selected[idx] * selected[other_idx]).mean())
        if not penalties:
            return matched_masks.sum() * 0.0
        return torch.stack(penalties).mean()

    def _single_output_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: Sequence[dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Sequence[Tuple[torch.Tensor, torch.Tensor]]]:
        indices = self.matcher(outputs, targets)
        loss_cls = self._classification_loss(outputs["pred_logits"], targets, indices)
        pred_boxes, target_boxes, pred_masks, target_masks, target_labels = self._matched_outputs(
            outputs,
            targets,
            indices,
        )

        if pred_boxes.numel() == 0:
            zero = outputs["pred_logits"].sum() * 0.0
            losses = {
                "loss_cls": loss_cls,
                "loss_bbox": zero,
                "loss_giou": zero,
                "loss_mask_bce": zero,
                "loss_mask_dice": zero,
                "loss_repulsion": zero,
            }
        else:
            loss_bbox = F.l1_loss(pred_boxes, target_boxes)
            giou = generalized_box_iou(pred_boxes, target_boxes)
            loss_giou = 1.0 - torch.diagonal(giou).mean()
            loss_mask_bce = _balanced_bce_with_logits(
                pred_masks.unsqueeze(1),
                target_masks.unsqueeze(1),
            )
            loss_mask_dice = _dice_loss(pred_masks.unsqueeze(1), target_masks.unsqueeze(1)).mean()
            loss_repulsion = self._repulsion_loss(pred_masks, target_labels)
            losses = {
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
                "loss_giou": loss_giou,
                "loss_mask_bce": loss_mask_bce,
                "loss_mask_dice": loss_mask_dice,
                "loss_repulsion": loss_repulsion,
            }

        total = (
            self.loss_weights["cls"] * losses["loss_cls"]
            + self.loss_weights["bbox"] * losses["loss_bbox"]
            + self.loss_weights["giou"] * losses["loss_giou"]
            + self.loss_weights["mask_bce"] * losses["loss_mask_bce"]
            + self.loss_weights["mask_dice"] * losses["loss_mask_dice"]
            + self.loss_weights["repulsion"] * losses["loss_repulsion"]
        )
        losses["loss_total"] = total
        return losses, indices

    def forward(
        self,
        outputs: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]],
        targets: Sequence[dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        main_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
            "pred_masks": outputs["pred_masks"],
        }
        main_losses, _ = self._single_output_losses(main_outputs, targets)
        results = dict(main_losses)

        total_loss = main_losses["loss_total"]
        aux_total = total_loss.new_zeros(())
        aux_outputs = outputs.get("aux_outputs", [])
        for aux_index, aux_output in enumerate(aux_outputs):
            aux_losses, _ = self._single_output_losses(aux_output, targets)
            scaled_aux = self.aux_loss_weight * aux_losses["loss_total"]
            aux_total = aux_total + scaled_aux
            for key, value in aux_losses.items():
                results[f"aux_{aux_index}_{key}"] = value

        results["loss_aux_total"] = aux_total
        results["loss"] = total_loss + aux_total
        return results
