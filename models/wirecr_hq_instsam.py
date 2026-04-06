"""Top-level WireCR-HQInstSAM model for the new mainline."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hq_mask_decoder import HQMaskDecoder
from .mask_nms import ClassWiseMaskNMS
from .pixel_decoder import WireCRPixelDecoder
from .prompt_builder_v2 import PromptBuilderV2
from .quality_head import QualityHead
from .query_instance_head import QueryInstanceHead
from .sam_backbone_v2 import SAMBackboneV2, normalize_sam_backbone_v2_config
from .score_fusion import ScoreFusion
from .wirecr_multiscale_adapter import WireCRMultiScaleAdapter

__all__ = ["WireCRHQInstSAM"]


class WireCRHQInstSAM(SAMBackboneV2):
    """Top-level model that owns the full coarse-to-refine instance pipeline."""

    PARAMETER_GROUP_KEYS = (
        "lora",
        "wirecr_adapter",
        "pixel_decoder",
        "query_head",
        "prompt_encoder",
        "hq_decoder",
    )

    def __init__(
        self,
        *,
        wirecr_out_channels: int = 256,
        wirecr_c2_adapter_size: str = "small",
        wirecr_c345_adapter_size: str = "medium",
        wirecr_compression_ratio: int = 8,
        wirecr_use_residual: bool = True,
        pixel_decoder_channels: int = 256,
        num_queries: int = 64,
        query_decoder_layers: int = 6,
        num_classes: int = 2,
        prompt_dense_prompt_downscale: int = 4,
        prompt_gt_box_jitter: float = 0.05,
        prompt_joint_gt_ratio_start: float = 0.7,
        prompt_joint_gt_ratio_end: float = 0.1,
        score_fusion_iou_threshold: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        sam_mask_decoder = self.hq_mask_decoder

        self.wirecr_multiscale_adapter = WireCRMultiScaleAdapter(
            in_channels={feature_name: self.embed_dim for feature_name in self.feature_names},
            out_channels=wirecr_out_channels,
            c2_adapter_size=wirecr_c2_adapter_size,
            c345_adapter_size=wirecr_c345_adapter_size,
            compression_ratio=wirecr_compression_ratio,
            use_residual=wirecr_use_residual,
        )
        self.pixel_decoder = WireCRPixelDecoder(
            in_channels=wirecr_out_channels,
            out_channels=pixel_decoder_channels,
        )
        self.query_head = QueryInstanceHead(
            hidden_dim=pixel_decoder_channels,
            num_classes=num_classes,
            num_queries=num_queries,
            decoder_layers=query_decoder_layers,
        )
        self.prompt_builder = PromptBuilderV2(
            dense_prompt_downscale=prompt_dense_prompt_downscale,
            gt_box_jitter=prompt_gt_box_jitter,
            joint_gt_ratio_start=prompt_joint_gt_ratio_start,
            joint_gt_ratio_end=prompt_joint_gt_ratio_end,
        )
        self.hq_mask_decoder = HQMaskDecoder(
            sam_mask_decoder,
            in_channels=wirecr_out_channels,
            prompt_dim=self.prompt_encoder.embed_dim,
        )
        self.quality_head = QualityHead(self.hq_mask_decoder.refine_feature_channels)
        self.score_fusion = ScoreFusion()
        self.mask_nms = ClassWiseMaskNMS(iou_threshold=score_fusion_iou_threshold)

        for parameter in self.hq_mask_decoder.parameters():
            parameter.requires_grad = True
        for parameter in self.quality_head.parameters():
            parameter.requires_grad = True
        for parameter in self.score_fusion.parameters():
            parameter.requires_grad = True

        self.num_queries = num_queries
        self.query_decoder_layers = query_decoder_layers
        self.num_classes = num_classes

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, object], **overrides) -> "WireCRHQInstSAM":
        raw_config = dict(model_config)
        decoder_layers = raw_config.pop("decoder_layers", None)
        if decoder_layers is not None and "query_decoder_layers" not in raw_config:
            raw_config["query_decoder_layers"] = decoder_layers
        normalized = normalize_sam_backbone_v2_config(raw_config)
        passthrough_keys = {
            "wirecr_out_channels",
            "wirecr_c2_adapter_size",
            "wirecr_c345_adapter_size",
            "wirecr_compression_ratio",
            "wirecr_use_residual",
            "pixel_decoder_channels",
            "num_queries",
            "query_decoder_layers",
            "num_classes",
            "prompt_dense_prompt_downscale",
            "prompt_gt_box_jitter",
            "prompt_joint_gt_ratio_start",
            "prompt_joint_gt_ratio_end",
            "score_fusion_iou_threshold",
        }
        for key in passthrough_keys:
            if key in raw_config:
                normalized[key] = raw_config[key]
        normalized.update(overrides)
        return cls(**normalized)

    def forward_multiscale_features(self, images: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor] | torch.Tensor]:
        backbone_outputs = self.forward_backbone(images)
        backbone_features = {feature_name: backbone_outputs[feature_name] for feature_name in self.feature_names}
        adapted_features = self.wirecr_multiscale_adapter(backbone_features)
        return {
            **backbone_outputs,
            "adapted_features": adapted_features,
        }

    def forward_pixel_decoder(self, images: torch.Tensor) -> Dict[str, object]:
        multiscale_outputs = self.forward_multiscale_features(images)
        pixel_outputs = self.pixel_decoder(multiscale_outputs["adapted_features"])
        return {
            **multiscale_outputs,
            **pixel_outputs,
        }

    def forward_query_head(self, images: torch.Tensor) -> Dict[str, object]:
        pixel_outputs = self.forward_pixel_decoder(images)
        query_outputs = self.query_head(
            pixel_outputs["mask_features"],
            pixel_outputs["multi_scale_memory"],
        )
        return {
            **pixel_outputs,
            **query_outputs,
        }

    def forward_coarse(self, images: torch.Tensor) -> Dict[str, object]:
        return self.forward_query_head(images)

    def _default_processed_sizes(self, batch_size: int) -> List[tuple[int, int]]:
        return [(int(self.image_size), int(self.image_size)) for _ in range(batch_size)]

    @staticmethod
    def _coarse_mask_score(mask_logits: torch.Tensor) -> torch.Tensor:
        return mask_logits.sigmoid().flatten(1).mean(dim=1)

    @staticmethod
    def _box_quality(pred_boxes: torch.Tensor) -> torch.Tensor:
        widths = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0.0)
        heights = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0.0)
        return (widths * heights).clamp(0.0, 1.0)

    def build_prompts(
        self,
        coarse_outputs: Mapping[str, torch.Tensor | Dict[str, torch.Tensor] | Sequence[torch.Tensor]],
        *,
        targets: Sequence[Mapping[str, torch.Tensor]] | None = None,
        processed_sizes: Sequence[tuple[int, int]] | None = None,
        prompt_source: str = "pred",
        joint_progress: float | None = None,
        gt_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Dict[str, object]:
        pred_logits = coarse_outputs["pred_logits"]
        pred_boxes = coarse_outputs["pred_boxes"]
        pred_masks = coarse_outputs["pred_masks"]
        batch_size = int(pred_logits.shape[0])
        if processed_sizes is None:
            processed_sizes = self._default_processed_sizes(batch_size)
        if len(processed_sizes) != batch_size:
            raise ValueError("processed_sizes length must match batch size.")

        prompt_batches = []
        coarse_instance_batches = []
        for batch_index in range(batch_size):
            logits = pred_logits[batch_index]
            boxes = pred_boxes[batch_index]
            masks = pred_masks[batch_index]

            probabilities = logits.softmax(dim=-1)
            labels = probabilities.argmax(dim=-1)
            foreground_mask = labels > 0
            selected_logits = logits[foreground_mask]
            selected_boxes = boxes[foreground_mask]
            selected_masks = masks[foreground_mask]
            selected_labels = labels[foreground_mask]
            if selected_logits.numel() == 0:
                class_scores = logits.new_zeros((0,))
            else:
                class_scores = probabilities[foreground_mask, selected_labels]

            coarse_instances = {
                "boxes": selected_boxes,
                "mask_logits": selected_masks,
                "labels": selected_labels,
                "class_scores": class_scores,
                "box_quality": self._box_quality(selected_boxes),
                "coarse_mask_score": self._coarse_mask_score(selected_masks),
                "processed_size": tuple(int(value) for value in processed_sizes[batch_index]),
                "query_indices": torch.nonzero(foreground_mask, as_tuple=False).flatten(),
                "oriented_boxes": selected_boxes.new_zeros((selected_boxes.shape[0], 5)),
                "principal_axes": selected_boxes.new_zeros((selected_boxes.shape[0], 2)),
            }
            coarse_instance_batches.append(coarse_instances)

            gt_instances = None
            if targets is not None:
                gt_instances = dict(targets[batch_index])
                gt_instances["processed_size"] = tuple(
                    int(value) for value in gt_instances.get("processed_size", processed_sizes[batch_index])
                )

            prompt_batches.append(
                self.prompt_builder.build_prompts(
                    pred_instances=coarse_instances,
                    gt_instances=gt_instances,
                    processed_size=tuple(int(value) for value in processed_sizes[batch_index]),
                    prompt_source=prompt_source,
                    joint_progress=joint_progress,
                    gt_ratio=gt_ratio,
                    generator=generator,
                )
            )

        return {
            "prompt_batches": prompt_batches,
            "coarse_instance_batches": coarse_instance_batches,
        }

    def forward_prompt_builder(
        self,
        images: torch.Tensor,
        *,
        targets: Sequence[Mapping[str, torch.Tensor]] | None = None,
        processed_sizes: Sequence[tuple[int, int]] | None = None,
        prompt_source: str = "pred",
        joint_progress: float | None = None,
        gt_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Dict[str, object]:
        coarse_outputs = self.forward_coarse(images)
        prompt_outputs = self.build_prompts(
            coarse_outputs,
            targets=targets,
            processed_sizes=processed_sizes,
            prompt_source=prompt_source,
            joint_progress=joint_progress,
            gt_ratio=gt_ratio,
            generator=generator,
        )
        return {
            **coarse_outputs,
            **prompt_outputs,
        }

    def forward_refine(
        self,
        images: torch.Tensor,
        *,
        coarse_outputs: Mapping[str, object] | None = None,
        prompt_batches: Sequence[Mapping[str, object]] | None = None,
        coarse_instance_batches: Sequence[Mapping[str, torch.Tensor]] | None = None,
        targets: Sequence[Mapping[str, torch.Tensor]] | None = None,
        processed_sizes: Sequence[tuple[int, int]] | None = None,
        prompt_source: str = "pred",
        joint_progress: float | None = None,
        gt_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Dict[str, object]:
        if coarse_outputs is None:
            coarse_outputs = self.forward_coarse(images)
        batch_size = int(images.shape[0])
        if processed_sizes is None:
            processed_sizes = self._default_processed_sizes(batch_size)
        if prompt_batches is None or coarse_instance_batches is None:
            prompt_outputs = self.build_prompts(
                coarse_outputs,
                targets=targets,
                processed_sizes=processed_sizes,
                prompt_source=prompt_source,
                joint_progress=joint_progress,
                gt_ratio=gt_ratio,
                generator=generator,
            )
            prompt_batches = prompt_outputs["prompt_batches"]
            coarse_instance_batches = prompt_outputs["coarse_instance_batches"]

        refine_batches = []
        image_pe = self.prompt_encoder.get_dense_pe()
        adapted_features = coarse_outputs["adapted_features"]
        for batch_index in range(batch_size):
            prompt_batch = prompt_batches[batch_index]
            coarse_batch = coarse_instance_batches[batch_index]
            boxes_xyxy = prompt_batch["boxes_xyxy"]
            dense_mask_prompt_logits = prompt_batch["dense_mask_prompt_logits"]
            point_coords = prompt_batch["point_coords"]
            point_labels = prompt_batch["point_labels"]
            prompt_meta = prompt_batch["prompt_meta"]

            if int(boxes_xyxy.shape[0]) == 0:
                empty_logits = images.new_zeros((0, 1, processed_sizes[batch_index][0], processed_sizes[batch_index][1]))
                refine_batches.append(
                    {
                        "boxes_xyxy": boxes_xyxy,
                        "labels": boxes_xyxy.new_zeros((0,), dtype=torch.long),
                        "coarse_mask_logits": empty_logits,
                        "refined_mask_logits": empty_logits,
                        "refine_features": images.new_zeros((0, self.hq_mask_decoder.refine_feature_channels, *empty_logits.shape[-2:])),
                        "decoder_scores": images.new_zeros((0,)),
                        "quality_scores": images.new_zeros((0,)),
                        "class_scores": images.new_zeros((0,)),
                        "box_quality": images.new_zeros((0,)),
                        "coarse_mask_score": images.new_zeros((0,)),
                        "prompt_meta": prompt_meta,
                    }
                )
                continue

            sparse_prompt_embeddings, dense_prompt_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=boxes_xyxy,
                masks=dense_mask_prompt_logits,
            )
            decoder_outputs = self.hq_mask_decoder(
                final_features=adapted_features["c5"][batch_index : batch_index + 1],
                early_features=adapted_features["c2"][batch_index : batch_index + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                dense_prompt_logits=dense_mask_prompt_logits,
            )

            prompt_labels = torch.as_tensor(
                [int(item["label"]) for item in prompt_meta],
                dtype=torch.long,
                device=boxes_xyxy.device,
            )
            source_query_indices = []
            coarse_query_indices = coarse_batch["query_indices"]
            for item in prompt_meta:
                if item.get("instance_source") == "pred":
                    source_index = int(item.get("source_index", -1))
                    if 0 <= source_index < int(coarse_query_indices.shape[0]):
                        source_query_indices.append(int(coarse_query_indices[source_index].item()))
                    else:
                        source_query_indices.append(-1)
                else:
                    source_query_indices.append(-1)
            count = int(boxes_xyxy.shape[0])
            class_scores = coarse_batch["class_scores"]
            box_quality = coarse_batch["box_quality"]
            coarse_mask_score = coarse_batch["coarse_mask_score"]
            if class_scores.shape[0] < count:
                pad_size = count - class_scores.shape[0]
                class_scores = torch.cat((class_scores, boxes_xyxy.new_full((pad_size,), 0.5)), dim=0)
                box_quality = torch.cat((box_quality, boxes_xyxy.new_ones((pad_size,))), dim=0)
                coarse_mask_score = torch.cat((coarse_mask_score, boxes_xyxy.new_full((pad_size,), 0.5)), dim=0)
            else:
                class_scores = class_scores[:count]
                box_quality = box_quality[:count]
                coarse_mask_score = coarse_mask_score[:count]

            quality_scores = self.quality_head(
                decoder_outputs["refine_features"],
                coarse_score=coarse_mask_score,
                decoder_score=decoder_outputs["decoder_scores"],
            )
            refine_batches.append(
                {
                    "boxes_xyxy": boxes_xyxy,
                    "labels": prompt_labels,
                    "coarse_mask_logits": decoder_outputs["coarse_mask_logits"],
                    "refined_mask_logits": decoder_outputs["refined_mask_logits"],
                    "refine_features": decoder_outputs["refine_features"],
                    "decoder_scores": decoder_outputs["decoder_scores"],
                    "quality_scores": quality_scores,
                    "class_scores": class_scores,
                    "box_quality": box_quality,
                    "coarse_mask_score": coarse_mask_score,
                    "source_query_indices": torch.as_tensor(
                        source_query_indices,
                        dtype=torch.long,
                        device=boxes_xyxy.device,
                    ),
                    "prompt_meta": prompt_meta,
                }
            )

        return {
            "refine_batches": refine_batches,
        }

    def fuse_scores(self, refine_outputs: Mapping[str, object]) -> Dict[str, object]:
        fused_batches = []
        for batch in refine_outputs["refine_batches"]:
            if int(batch["labels"].numel()) == 0:
                fused_batches.append({**batch, "instance_scores": batch["quality_scores"]})
                continue
            instance_scores = self.score_fusion(
                class_score=batch["class_scores"],
                box_quality=batch["box_quality"],
                coarse_mask_score=batch["coarse_mask_score"],
                refine_quality_score=batch["quality_scores"],
            )
            fused_batches.append({**batch, "instance_scores": instance_scores})
        return {
            "fused_batches": fused_batches,
        }

    def postprocess_instances(
        self,
        fused_outputs: Mapping[str, object],
        *,
        score_threshold: float = 0.05,
    ) -> Dict[str, object]:
        instances = []
        for batch in fused_outputs["fused_batches"]:
            if int(batch["labels"].numel()) == 0:
                instances.append([])
                continue

            refined_logits = batch["refined_mask_logits"][:, 0]
            binary_masks = refined_logits > 0
            keep_indices = self.mask_nms(
                binary_masks,
                batch["instance_scores"],
                batch["labels"],
            )
            sample_instances = []
            for keep_index in keep_indices:
                score = float(batch["instance_scores"][keep_index].item())
                if score < score_threshold:
                    continue
                sample_instances.append(
                    {
                        "label": int(batch["labels"][keep_index].item()),
                        "box": batch["boxes_xyxy"][keep_index].detach().cpu(),
                        "mask_logits": batch["refined_mask_logits"][keep_index, 0].detach().cpu(),
                        "mask": binary_masks[keep_index].detach().cpu(),
                        "instance_score": score,
                        "quality_score": float(batch["quality_scores"][keep_index].item()),
                        "class_score": float(batch["class_scores"][keep_index].item()),
                        "prompt_meta": batch["prompt_meta"][keep_index],
                    }
                )
            instances.append(sample_instances)
        return {
            "instances": instances,
        }

    def forward(
        self,
        images: torch.Tensor,
        *,
        targets: Sequence[Mapping[str, torch.Tensor]] | None = None,
        processed_sizes: Sequence[tuple[int, int]] | None = None,
        prompt_source: str | None = None,
        joint_progress: float | None = None,
        gt_ratio: float | None = None,
        generator: torch.Generator | None = None,
        score_threshold: float = 0.05,
    ) -> Dict[str, object]:
        if prompt_source is None:
            prompt_source = "gt" if targets is not None and self.training else "pred"

        coarse_outputs = self.forward_coarse(images)
        prompt_outputs = self.build_prompts(
            coarse_outputs,
            targets=targets,
            processed_sizes=processed_sizes,
            prompt_source=prompt_source,
            joint_progress=joint_progress,
            gt_ratio=gt_ratio,
            generator=generator,
        )
        refine_outputs = self.forward_refine(
            images,
            coarse_outputs=coarse_outputs,
            prompt_batches=prompt_outputs["prompt_batches"],
            coarse_instance_batches=prompt_outputs["coarse_instance_batches"],
            targets=targets,
            processed_sizes=processed_sizes,
            prompt_source=prompt_source,
            joint_progress=joint_progress,
            gt_ratio=gt_ratio,
            generator=generator,
        )
        fused_outputs = self.fuse_scores(refine_outputs)
        postprocessed = self.postprocess_instances(fused_outputs, score_threshold=score_threshold)

        return {
            "training_dict": {
                "coarse_outputs": coarse_outputs,
                "prompt_batches": prompt_outputs["prompt_batches"],
                "coarse_instance_batches": prompt_outputs["coarse_instance_batches"],
                "refine_batches": refine_outputs["refine_batches"],
            },
            "eval_dict": {
                "fused_batches": fused_outputs["fused_batches"],
                "instances": postprocessed["instances"],
            },
            "inference_dict": {
                "instances": postprocessed["instances"],
            },
        }

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups = {key: [] for key in self.PARAMETER_GROUP_KEYS}
        groups["lora"] = list(self.iter_lora_parameters())
        groups["wirecr_adapter"] = [
            parameter for parameter in self.wirecr_multiscale_adapter.parameters() if parameter.requires_grad
        ]
        groups["pixel_decoder"] = [
            parameter for parameter in self.pixel_decoder.parameters() if parameter.requires_grad
        ]
        groups["query_head"] = [
            parameter for parameter in self.query_head.parameters() if parameter.requires_grad
        ]
        groups["prompt_encoder"] = [
            parameter for parameter in self.prompt_encoder.parameters() if parameter.requires_grad
        ]
        groups["hq_decoder"] = [
            parameter
            for module in (self.hq_mask_decoder, self.quality_head, self.score_fusion)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return groups

    def get_trainable_parameter_report(self) -> Dict[str, Dict[str, int]]:
        groups = self.get_parameter_groups()
        report: Dict[str, Dict[str, int]] = {}
        total_trainable = 0
        for group_name, parameters in groups.items():
            parameter_count = sum(parameter.numel() for parameter in parameters)
            tensor_count = len(parameters)
            report[group_name] = {
                "parameter_count": parameter_count,
                "tensor_count": tensor_count,
            }
            total_trainable += parameter_count

        report["summary"] = {
            "parameter_count": total_trainable,
            "tensor_count": sum(item["tensor_count"] for item in report.values()),
        }
        return report
