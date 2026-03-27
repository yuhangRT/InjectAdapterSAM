"""
Per-instance SAM refinement with multimask selection.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["SAMInstanceRefiner"]


def _mask_in_box_recall(mask: torch.Tensor, box: torch.Tensor) -> float:
    x1, y1, x2, y2 = box.round().long().tolist()
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, mask.shape[-1])
    y2 = min(y2, mask.shape[-2])
    total = float((mask > 0).sum().item())
    if total <= 0:
        return 0.0
    inside = float((mask[..., y1:y2, x1:x2] > 0).sum().item())
    return inside / max(total, 1.0)


class SAMInstanceRefiner(nn.Module):
    def __init__(self, backend, inside_box_recall_threshold: float = 0.70):
        super().__init__()
        self.backend = backend
        self.inside_box_recall_threshold = float(inside_box_recall_threshold)

    def forward(
        self,
        *,
        image_embeddings: torch.Tensor,
        prompt_batch: dict[str, torch.Tensor | list[str]],
        processed_size: tuple[int, int],
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        if prompt_batch["boxes"].numel() == 0:
            empty_masks = torch.zeros(
                (0, 1, output_size[0], output_size[1]) if output_size else (0, 1, processed_size[0], processed_size[1]),
                device=image_embeddings.device,
            )
            return {
                "selected_masks": empty_masks,
                "selected_masks_processed": torch.zeros((0, 1, processed_size[0], processed_size[1]), device=image_embeddings.device),
                "selected_scores": torch.zeros((0,), device=image_embeddings.device),
                "selected_indices": torch.zeros((0,), dtype=torch.long, device=image_embeddings.device),
                "low_res_masks": torch.zeros((0, 3, 256, 256), device=image_embeddings.device),
                "iou_scores": torch.zeros((0, 3), device=image_embeddings.device),
                "source_prompt": [],
            }

        sparse_embeddings, dense_embeddings = self.backend.encode_prompts(
            points=(prompt_batch["point_coords"], prompt_batch["point_labels"]),
            boxes=prompt_batch["boxes"],
            masks=None,
        )
        low_res_masks, iou_scores = self.backend.decode_masks(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        processed_candidates = self.backend.postprocess_masks(
            low_res_masks,
            input_size=processed_size,
            original_size=processed_size,
        )
        target_output_size = output_size or processed_size
        selected_logits = []
        selected_scores = []
        selected_indices = []

        for prompt_idx in range(processed_candidates.shape[0]):
            candidate_logits = processed_candidates[prompt_idx]
            candidate_scores = iou_scores[prompt_idx]
            box = prompt_batch["boxes"][prompt_idx]
            valid_indices = []
            for candidate_idx in range(candidate_logits.shape[0]):
                recall = _mask_in_box_recall(candidate_logits[candidate_idx : candidate_idx + 1], box)
                if recall >= self.inside_box_recall_threshold:
                    valid_indices.append(candidate_idx)
            if not valid_indices:
                valid_indices = list(range(candidate_logits.shape[0]))
            best_idx = max(valid_indices, key=lambda idx: float(candidate_scores[idx].item()))
            selected_indices.append(best_idx)
            selected_scores.append(candidate_scores[best_idx])
            selected_logits.append(low_res_masks[prompt_idx : prompt_idx + 1, best_idx : best_idx + 1])

        selected_low_res = torch.cat(selected_logits, dim=0)
        selected_masks_processed = self.backend.postprocess_masks(
            selected_low_res,
            input_size=processed_size,
            original_size=processed_size,
        )
        selected_masks = self.backend.postprocess_masks(selected_low_res, input_size=processed_size, original_size=target_output_size)
        return {
            "selected_masks": selected_masks,
            "selected_masks_processed": selected_masks_processed,
            "selected_scores": torch.stack(selected_scores, dim=0),
            "selected_indices": torch.tensor(selected_indices, device=selected_masks.device, dtype=torch.long),
            "low_res_masks": low_res_masks,
            "iou_scores": iou_scores,
            "source_prompt": prompt_batch["source_prompt"],
        }
