"""
Legacy WireCR-InstSAM V1 model.

This file is historical only. New mainline code must not import it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbones.sam_wirecr_backbone import SAMWireCRBackbone
from models.heads.instance_proposal_head import CenterNetLiteProposalHead
from models.prompts.instance_prompt_builder import InstancePromptBuilder
from models.refine.roi_boundary_refiner import ROIBoundaryRefiner
from models.refine.sam_instance_refiner import SAMInstanceRefiner

__all__ = ["WireCRInstSAM"]


class WireCRInstSAM(nn.Module):
    def __init__(
        self,
        backend,
        *,
        freeze_encoder: bool = True,
        enable_roi_refiner: bool = False,
        topk_per_class: int = 64,
        box_nms_iou: float = 0.5,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.backbone = SAMWireCRBackbone(backend, freeze_encoder=freeze_encoder)
        self.proposal_head = CenterNetLiteProposalHead(in_channels=256, feat_channels=128, num_classes=2, stride=8)
        self.prompt_builder = InstancePromptBuilder(expand_ratio=1.10)
        self.refiner = SAMInstanceRefiner(backend, inside_box_recall_threshold=0.70)
        self.roi_refiner = ROIBoundaryRefiner(feature_channels=256, hidden_channels=64)
        self.enable_roi_refiner = bool(enable_roi_refiner)
        self.topk_per_class = int(topk_per_class)
        self.box_nms_iou = float(box_nms_iou)

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.backend.preprocess(image.float()) for image in images], dim=0)

    def forward_backbone(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        preprocessed = self.preprocess_images(images.to(self.backend.device))
        features = self.backbone(preprocessed)
        features["preprocessed_images"] = preprocessed
        return features

    def forward_proposals(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.proposal_head(features)

    @torch.no_grad()
    def decode_proposals(self, proposal_outputs: dict[str, torch.Tensor], *, image_size: int) -> list[list[dict[str, Any]]]:
        return self.proposal_head.decode(
            proposal_outputs,
            image_size=image_size,
            topk_per_class=self.topk_per_class,
            box_nms_iou=self.box_nms_iou,
        )

    def build_gt_proposals(self, instances: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        batch_proposals = []
        for sample_instances in instances:
            sample_proposals = []
            for box, label, mask in zip(sample_instances["boxes"], sample_instances["labels"], sample_instances["masks"]):
                sample_proposals.append(
                    {
                        "bbox": box.to(self.backend.device),
                        "category_id": int(label.item()),
                        "score": 1.0,
                        "gt_mask": mask.to(self.backend.device),
                    }
                )
            batch_proposals.append(sample_proposals)
        return batch_proposals

    def refine_instances(
        self,
        *,
        features: dict[str, torch.Tensor],
        proposals: list[list[dict[str, Any]]],
        processed_sizes: list[tuple[int, int]],
        output_sizes: list[tuple[int, int]] | None = None,
        apply_roi_refiner: bool = False,
    ) -> list[dict[str, Any]]:
        image_embeddings = features["image_embeddings"]
        low_level_features = features["c2"]
        results = []
        for batch_idx, sample_proposals in enumerate(proposals):
            processed_size = processed_sizes[batch_idx]
            output_size = processed_size if output_sizes is None else output_sizes[batch_idx]
            prompt_batch = self.prompt_builder.build_batch_prompts(sample_proposals, image_size=processed_size)
            prompt_batch = {
                key: value.to(image_embeddings.device) if torch.is_tensor(value) else value
                for key, value in prompt_batch.items()
            }
            refined = self.refiner(
                image_embeddings=image_embeddings[batch_idx : batch_idx + 1],
                prompt_batch=prompt_batch,
                processed_size=processed_size,
                output_size=output_size,
            )
            selected_masks = refined["selected_masks"]
            if apply_roi_refiner and self.enable_roi_refiner and selected_masks.numel() > 0 and output_size == processed_size:
                low_level = low_level_features[batch_idx : batch_idx + 1]
                refined_masks = []
                for mask_logit in selected_masks:
                    refined_masks.append(self.roi_refiner(mask_logit.unsqueeze(0), low_level)[0])
                selected_masks = torch.stack(refined_masks, dim=0)
            sample_results = []
            for proposal_idx, proposal in enumerate(sample_proposals):
                sample_results.append(
                    {
                        "instance_id": proposal_idx,
                        "category_id": int(proposal["category_id"]),
                        "score": float(refined["selected_scores"][proposal_idx].item() * proposal["score"]),
                        "bbox": proposal["bbox"].detach().clone(),
                        "mask_logits": selected_masks[proposal_idx, 0],
                        "mask_logits_processed": refined["selected_masks_processed"][proposal_idx, 0],
                        "mask": (selected_masks[proposal_idx, 0] > 0).float(),
                        "mask_processed": (refined["selected_masks_processed"][proposal_idx, 0] > 0).float(),
                        "source_prompt": refined["source_prompt"][proposal_idx],
                    }
                )
            results.append({"instances": sample_results, "refine": refined})
        return results
