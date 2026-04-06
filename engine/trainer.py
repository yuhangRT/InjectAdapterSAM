"""Trainer for the WireCR-HQInstSAM unified training pipeline."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import amp as torch_amp

from utils.checkpoint import DEFAULT_CHECKPOINT_FILENAMES, build_checkpoint_state, load_checkpoint, save_checkpoint
from utils.losses_coarse import CoarseLossCriterion
from utils.losses_refine import RefineLossCriterion

__all__ = [
    "CurriculumState",
    "TrainerState",
    "WireCRHQInstSAMTrainer",
]


@dataclass(frozen=True)
class CurriculumState:
    epoch: int
    prompt_source: str
    gt_ratio: float
    refine_loss_boost: float


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metrics: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.best_metrics is None:
            self.best_metrics = {
                "mask_ap": float("-inf"),
                "mask_ap50": float("-inf"),
                "empty_terminal_recall": float("-inf"),
            }


class _ProxyMetricAccumulator:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tp_hole = 0
        self.gt_hole = 0
        self.matched_iou_sum = 0.0
        self.count = 0

    @staticmethod
    def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
        mask_a = mask_a.bool()
        if mask_a.shape != mask_b.shape:
            mask_b = F.interpolate(
                mask_b.unsqueeze(0).unsqueeze(0).float(),
                size=mask_a.shape[-2:],
                mode="nearest",
            )[0, 0] > 0.5
        else:
            mask_b = mask_b.bool()
        intersection = float((mask_a & mask_b).float().sum().item())
        union = float((mask_a | mask_b).float().sum().item())
        if union <= 0.0:
            return 0.0
        return intersection / union

    def update(self, predictions: Sequence[dict[str, Any]], target: Mapping[str, torch.Tensor]) -> None:
        labels = target["labels"].detach().cpu()
        masks = target["masks"].detach().cpu().bool()
        for class_label in (1, 2):
            pred_items = [item for item in predictions if int(item["label"]) == class_label]
            pred_items.sort(key=lambda item: float(item["instance_score"]), reverse=True)
            gt_indices = [idx for idx, label in enumerate(labels.tolist()) if int(label) == class_label]
            matched_gt: set[int] = set()

            if class_label == 2:
                self.gt_hole += len(gt_indices)

            for pred in pred_items:
                best_iou = 0.0
                best_gt = None
                pred_mask = pred["mask"]
                for gt_index in gt_indices:
                    if gt_index in matched_gt:
                        continue
                    iou = self._mask_iou(pred_mask, masks[gt_index])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_index
                if best_gt is not None and best_iou >= 0.5:
                    matched_gt.add(best_gt)
                    self.tp += 1
                    self.matched_iou_sum += best_iou
                    self.count += 1
                    if class_label == 2:
                        self.tp_hole += 1
                else:
                    self.fp += 1

            self.fn += max(len(gt_indices) - len(matched_gt), 0)

    def summarize(self) -> dict[str, float]:
        denom = max(self.tp + self.fp + self.fn, 1)
        return {
            "mask_ap": float(self.matched_iou_sum / denom),
            "mask_ap50": float(self.tp / denom),
            "empty_terminal_recall": float(self.tp_hole / max(self.gt_hole, 1)),
        }


class WireCRHQInstSAMTrainer:
    """Single-trainer implementation with warmup/joint curriculum and checkpointing."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        output_dir: str | Path,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Iterable[Mapping[str, Any]] | None = None,
        device: str | torch.device = "cpu",
        epochs: int = 60,
        warmup_epochs: int = 8,
        lr_lora: float = 5e-5,
        lr_new_modules: float = 2e-4,
        weight_decay: float = 0.05,
        warmup_iters: int = 1000,
        grad_accum_steps: int = 2,
        max_grad_norm: float = 0.1,
        val_interval: int = 1,
        amp: bool = True,
        coarse_criterion: CoarseLossCriterion | None = None,
        refine_criterion: RefineLossCriterion | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        scaler: torch_amp.GradScaler | None = None,
        evaluator: Any | None = None,
        config_snapshot: Mapping[str, Any] | None = None,
        infer_config: Mapping[str, Any] | None = None,
        checkpoint_filenames: Mapping[str, str] | None = None,
        resume_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.lr_lora = float(lr_lora)
        self.lr_new_modules = float(lr_new_modules)
        self.weight_decay = float(weight_decay)
        self.warmup_iters = int(warmup_iters)
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.max_grad_norm = float(max_grad_norm)
        self.val_interval = max(int(val_interval), 1)
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        self.coarse_criterion = coarse_criterion or CoarseLossCriterion()
        self.refine_criterion = refine_criterion or RefineLossCriterion()
        self.config_snapshot = copy.deepcopy(dict(config_snapshot or {}))
        self.infer_config = dict(infer_config or {})
        self.checkpoint_filenames = dict(DEFAULT_CHECKPOINT_FILENAMES)
        if checkpoint_filenames is not None:
            self.checkpoint_filenames.update(checkpoint_filenames)

        self.model.to(self.device)
        self.coarse_criterion.to(self.device)
        self.refine_criterion.to(self.device)

        self.optimizer = optimizer or self.build_optimizer()
        self.scheduler = scheduler or self.build_scheduler()
        self.scaler = scaler or torch_amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.evaluator = evaluator
        self.state = TrainerState()
        self.history: list[dict[str, Any]] = []

        if resume_path is not None:
            self.resume(resume_path)

    def build_optimizer(self) -> torch.optim.Optimizer:
        parameter_groups = self.model.get_parameter_groups()
        optimizer_groups = []
        if parameter_groups["lora"]:
            optimizer_groups.append(
                {
                    "params": parameter_groups["lora"],
                    "lr": self.lr_lora,
                    "weight_decay": self.weight_decay,
                    "group_name": "lora",
                }
            )
        for group_name in ("wirecr_adapter", "pixel_decoder", "query_head", "prompt_encoder", "hq_decoder"):
            params = parameter_groups[group_name]
            if not params:
                continue
            optimizer_groups.append(
                {
                    "params": params,
                    "lr": self.lr_new_modules,
                    "weight_decay": self.weight_decay,
                    "group_name": group_name,
                }
            )
        if not optimizer_groups:
            raise ValueError("No trainable parameter groups found for optimizer construction.")
        return torch.optim.AdamW(optimizer_groups, lr=self.lr_new_modules, weight_decay=self.weight_decay)

    def build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        train_steps = max(len(self.train_loader), 1)
        total_optimizer_steps = math.ceil(train_steps * max(self.epochs, 1) / self.grad_accum_steps)
        warmup_steps = min(max(self.warmup_iters, 0), total_optimizer_steps)

        def _schedule(step: int) -> float:
            if total_optimizer_steps <= 0:
                return 1.0
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if total_optimizer_steps == warmup_steps:
                return 1.0
            progress = float(step - warmup_steps) / float(max(total_optimizer_steps - warmup_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, _schedule)

    def resolve_curriculum(self, epoch: int) -> CurriculumState:
        if epoch < self.warmup_epochs:
            return CurriculumState(
                epoch=epoch,
                prompt_source="gt",
                gt_ratio=1.0,
                refine_loss_boost=1.5,
            )
        if self.epochs <= self.warmup_epochs:
            progress = 1.0
        else:
            progress = float(epoch - self.warmup_epochs) / float(max(self.epochs - self.warmup_epochs - 1, 1))
        gt_ratio = 0.7 + (0.1 - 0.7) * min(max(progress, 0.0), 1.0)
        return CurriculumState(
            epoch=epoch,
            prompt_source="mixed",
            gt_ratio=float(gt_ratio),
            refine_loss_boost=1.0,
        )

    @staticmethod
    def _to_device(
        targets: Sequence[Mapping[str, Any]],
        device: torch.device,
        processed_sizes: Sequence[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        moved: list[dict[str, Any]] = []
        for index, target in enumerate(targets):
            item: dict[str, Any] = {}
            for key, value in target.items():
                if torch.is_tensor(value):
                    item[key] = value.to(device)
                else:
                    item[key] = value
            if processed_sizes is not None and "processed_size" not in item:
                item["processed_size"] = tuple(int(value) for value in processed_sizes[index])
            moved.append(item)
        return moved

    def _build_refine_targets(
        self,
        refine_batches: Sequence[Mapping[str, Any]],
        coarse_outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        logits = []
        target_masks = []
        quality_scores = []
        coarse_main_outputs = {
            "pred_logits": coarse_outputs["pred_logits"],
            "pred_boxes": coarse_outputs["pred_boxes"],
            "pred_masks": coarse_outputs["pred_masks"],
        }
        coarse_indices = self.coarse_criterion.matcher(coarse_main_outputs, targets)
        for batch_index, refine_batch in enumerate(refine_batches):
            prompt_meta = refine_batch["prompt_meta"]
            if not prompt_meta:
                continue
            gt_masks = targets[batch_index]["masks"].to(self.device).float()
            query_to_gt = {
                int(query_idx.item()): int(gt_idx.item())
                for query_idx, gt_idx in zip(*coarse_indices[batch_index])
            }
            source_query_indices = refine_batch.get("source_query_indices")
            for prompt_index, meta in enumerate(prompt_meta):
                matched_gt_index = int(meta.get("matched_gt_index", -1))
                if matched_gt_index < 0 and source_query_indices is not None and prompt_index < int(source_query_indices.shape[0]):
                    source_query_index = int(source_query_indices[prompt_index].item())
                    matched_gt_index = query_to_gt.get(source_query_index, -1)
                if matched_gt_index < 0 or matched_gt_index >= gt_masks.shape[0]:
                    continue
                logits.append(refine_batch["refined_mask_logits"][prompt_index : prompt_index + 1])
                target_masks.append(gt_masks[matched_gt_index : matched_gt_index + 1])
                quality_scores.append(refine_batch["quality_scores"][prompt_index : prompt_index + 1])

        if not logits:
            return None, None

        stacked_logits = torch.cat(logits, dim=0)
        stacked_targets = torch.cat(target_masks, dim=0).unsqueeze(1)
        if stacked_targets.shape[-2:] != stacked_logits.shape[-2:]:
            stacked_targets = F.interpolate(stacked_targets, size=stacked_logits.shape[-2:], mode="nearest")
        stacked_quality = torch.cat(quality_scores, dim=0)
        return (
            stacked_logits,
            {"target_masks": stacked_targets, "quality_scores": stacked_quality},
        )

    def _compute_refine_losses(
        self,
        refine_batches: Sequence[Mapping[str, Any]],
        coarse_outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, torch.Tensor]:
        packed_logits, packed_targets = self._build_refine_targets(refine_batches, coarse_outputs, targets)
        if packed_logits is None or packed_targets is None:
            zero = next(self.model.parameters()).sum() * 0.0
            return {
                "loss_bce": zero,
                "loss_dice": zero,
                "loss_boundary": zero,
                "loss_quality": zero,
                "loss_total": zero,
                "loss": zero,
            }
        return self.refine_criterion(
            refined_mask_logits=packed_logits,
            target_masks=packed_targets["target_masks"],
            quality_scores=packed_targets["quality_scores"],
        )

    @staticmethod
    def _collect_prompt_source_stats(prompt_batches: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        total = 0
        gt_count = 0
        pred_count = 0
        for batch in prompt_batches:
            for meta in batch["prompt_meta"]:
                total += 1
                source = meta.get("instance_source")
                if source == "gt":
                    gt_count += 1
                elif source == "pred":
                    pred_count += 1
        if total == 0:
            return {
                "prompt_gt_fraction": 0.0,
                "prompt_pred_fraction": 0.0,
            }
        return {
            "prompt_gt_fraction": float(gt_count / total),
            "prompt_pred_fraction": float(pred_count / total),
        }

    def _default_thresholds(self) -> dict[str, Any]:
        score_thresh_label = self.infer_config.get("score_thresh_label", 0.5)
        score_thresh_hole = self.infer_config.get("score_thresh_hole", 0.5)
        def _coerce_threshold(value: Any) -> float:
            if isinstance(value, str) and value == "auto_from_val":
                return 0.5
            return float(value)

        return {
            "score_thresh_label": _coerce_threshold(score_thresh_label),
            "score_thresh_hole": _coerce_threshold(score_thresh_hole),
            "mask_nms_iou_label": float(getattr(self.model.mask_nms, "iou_threshold", 0.6)),
            "mask_nms_iou_hole": float(getattr(self.model.mask_nms, "iou_threshold", 0.6)),
        }

    def validate(self) -> dict[str, Any]:
        if self.evaluator is not None and self.val_loader is not None:
            evaluation = self.evaluator.evaluate(self.val_loader)
            best_thresholds = dict(evaluation["best_thresholds"])
            best_metrics = dict(evaluation["best_metrics"])
            if "mask_ap50" not in best_metrics and "AP50" in best_metrics:
                best_metrics["mask_ap50"] = float(best_metrics["AP50"])
            if "mask_ap75" not in best_metrics and "AP75" in best_metrics:
                best_metrics["mask_ap75"] = float(best_metrics["AP75"])
            best_metrics.setdefault("val_loss", 0.0)
            best_metrics["best_val_thresholds"] = best_thresholds
            return best_metrics
        if self.val_loader is None:
            return {
                "val_loss": 0.0,
                "mask_ap": 0.0,
                "mask_ap50": 0.0,
                "empty_terminal_recall": 0.0,
                "best_val_thresholds": self._default_thresholds(),
            }

        self.model.eval()
        metric_accumulator = _ProxyMetricAccumulator()
        loss_totals = {
            "coarse_loss": 0.0,
            "refine_loss": 0.0,
            "total_loss": 0.0,
        }
        batch_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                targets = self._to_device(
                    batch["instances"],
                    self.device,
                    processed_sizes=batch.get("processed_size"),
                )
                outputs = self.model(
                    images,
                    targets=targets,
                    processed_sizes=batch["processed_size"],
                    prompt_source="pred",
                )
                coarse_losses = self.coarse_criterion(outputs["training_dict"]["coarse_outputs"], targets)
                refine_losses = self._compute_refine_losses(
                    outputs["training_dict"]["refine_batches"],
                    outputs["training_dict"]["coarse_outputs"],
                    targets,
                )
                total_loss = coarse_losses["loss"] + refine_losses["loss"]

                loss_totals["coarse_loss"] += float(coarse_losses["loss"].item())
                loss_totals["refine_loss"] += float(refine_losses["loss"].item())
                loss_totals["total_loss"] += float(total_loss.item())
                batch_count += 1

                for sample_predictions, target in zip(outputs["inference_dict"]["instances"], targets):
                    metric_accumulator.update(sample_predictions, target)

        metrics = metric_accumulator.summarize()
        metrics.update(
            {
                "val_loss": float(loss_totals["total_loss"] / max(batch_count, 1)),
                "val_coarse_loss": float(loss_totals["coarse_loss"] / max(batch_count, 1)),
                "val_refine_loss": float(loss_totals["refine_loss"] / max(batch_count, 1)),
                "best_val_thresholds": self._default_thresholds(),
            }
        )
        return metrics

    def _optimizer_step(self) -> None:
        if self.max_grad_norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.state.global_step += 1

    def train_one_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        curriculum = self.resolve_curriculum(epoch)
        optimizer_steps = 0
        totals = {
            "coarse_loss": 0.0,
            "refine_loss": 0.0,
            "total_loss": 0.0,
        }
        prompt_stats = {
            "prompt_gt_fraction": 0.0,
            "prompt_pred_fraction": 0.0,
        }
        batch_count = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = self._to_device(
                batch["instances"],
                self.device,
                processed_sizes=batch.get("processed_size"),
            )
            with torch_amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(
                    images,
                    targets=targets,
                    processed_sizes=batch["processed_size"],
                    prompt_source=curriculum.prompt_source,
                    gt_ratio=curriculum.gt_ratio,
                    joint_progress=0.0 if curriculum.prompt_source == "gt" else curriculum.gt_ratio,
                )
                coarse_losses = self.coarse_criterion(outputs["training_dict"]["coarse_outputs"], targets)
                refine_losses = self._compute_refine_losses(
                    outputs["training_dict"]["refine_batches"],
                    outputs["training_dict"]["coarse_outputs"],
                    targets,
                )
                total_loss = coarse_losses["loss"] + curriculum.refine_loss_boost * refine_losses["loss"]

            scaled_loss = total_loss / self.grad_accum_steps
            self.scaler.scale(scaled_loss).backward()

            should_step = ((batch_index + 1) % self.grad_accum_steps == 0) or ((batch_index + 1) == len(self.train_loader))
            if should_step:
                self._optimizer_step()
                optimizer_steps += 1

            totals["coarse_loss"] += float(coarse_losses["loss"].item())
            totals["refine_loss"] += float(refine_losses["loss"].item())
            totals["total_loss"] += float(total_loss.item())
            source_stats = self._collect_prompt_source_stats(outputs["training_dict"]["prompt_batches"])
            prompt_stats["prompt_gt_fraction"] += source_stats["prompt_gt_fraction"]
            prompt_stats["prompt_pred_fraction"] += source_stats["prompt_pred_fraction"]
            batch_count += 1

        metrics = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train_coarse_loss": float(totals["coarse_loss"] / max(batch_count, 1)),
            "train_refine_loss": float(totals["refine_loss"] / max(batch_count, 1)),
            "train_total_loss": float(totals["total_loss"] / max(batch_count, 1)),
            "prompt_source": curriculum.prompt_source,
            "prompt_gt_ratio": float(curriculum.gt_ratio),
            "refine_loss_boost": float(curriculum.refine_loss_boost),
            "prompt_gt_fraction": float(prompt_stats["prompt_gt_fraction"] / max(batch_count, 1)),
            "prompt_pred_fraction": float(prompt_stats["prompt_pred_fraction"] / max(batch_count, 1)),
            "learning_rates": {group["group_name"]: group["lr"] for group in self.optimizer.param_groups},
        }
        return metrics

    def _save_checkpoint_bundle(self, val_metrics: Mapping[str, Any]) -> None:
        best_rules = {
            "best_ap": "mask_ap",
            "best_ap50": "mask_ap50",
            "best_hole_recall": "empty_terminal_recall",
        }
        improved = set()
        for checkpoint_key, metric_key in best_rules.items():
            metric_value = float(val_metrics.get(metric_key, float("-inf")))
            if metric_value >= float(self.state.best_metrics.get(metric_key, float("-inf"))):
                self.state.best_metrics[metric_key] = metric_value
                improved.add(checkpoint_key)

        state = build_checkpoint_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            best_metrics=self.state.best_metrics,
            val_metrics_summary=val_metrics,
            best_val_thresholds=val_metrics.get("best_val_thresholds", self._default_thresholds()),
            config_snapshot=self.config_snapshot,
        )
        save_checkpoint(self.output_dir / self.checkpoint_filenames["last"], state)
        for checkpoint_key in improved:
            save_checkpoint(self.output_dir / self.checkpoint_filenames[checkpoint_key], state)

    def _append_history(self, entry: Mapping[str, Any]) -> None:
        self.history.append(dict(entry))
        history_path = self.output_dir / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def fit(self) -> list[dict[str, Any]]:
        latest_val_metrics: dict[str, Any] = {
            "val_loss": 0.0,
            "mask_ap": 0.0,
            "mask_ap50": 0.0,
            "empty_terminal_recall": 0.0,
            "best_val_thresholds": self._default_thresholds(),
        }
        for epoch in range(self.state.epoch, self.epochs):
            train_metrics = self.train_one_epoch(epoch)
            should_validate = ((epoch + 1) % self.val_interval == 0) or ((epoch + 1) == self.epochs)
            if should_validate:
                latest_val_metrics = self.validate()
            self.state.epoch = epoch + 1
            summary = {**train_metrics, **latest_val_metrics}
            self._save_checkpoint_bundle(latest_val_metrics)
            self._append_history(summary)
        return self.history

    def resume(self, checkpoint_path: str | Path) -> dict[str, Any]:
        state = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.state.epoch = int(state.get("epoch", 0))
        self.state.global_step = int(state.get("global_step", 0))
        loaded_best = state.get("best_metrics", {})
        if loaded_best:
            self.state.best_metrics.update({key: float(value) for key, value in loaded_best.items()})
        return state
