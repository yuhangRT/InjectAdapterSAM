"""
Training entrypoint for WireCR-InstSAM.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.optim as optim

from dataset.wire_hole_instance_dataset import get_instance_dataloader
from utils import logger
from utils.init import init_device
from utils.instsam_cli import add_common_instsam_args
from utils.instsam_runtime import build_instsam_model, resolve_instsam_run_dir, save_checkpoint
from utils.instance_losses import CenterNetProposalLoss, InstanceRefineLoss
from utils.instance_matcher import greedy_match_boxes, proposal_recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WireCR-InstSAM.")
    add_common_instsam_args(parser)
    parser.add_argument("--phase", choices=["oracle", "proposal", "refine", "joint"], default="proposal")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--refine-gt-ratio", type=float, default=0.5, help="Fraction of refine epochs that use GT proposals.")
    parser.add_argument("--train-metrics-interval", type=int, default=1, help="Compute train recall metrics every N steps. Use 0 to disable.")
    return parser.parse_args()


def _prepare_loaders(args):
    train_loader = get_instance_dataloader(
        data_root=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        image_size=args.image_size,
        roi_prob=0.4,
        roi_focus_prob=0.7,
        seed=args.seed,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = get_instance_dataloader(
        data_root=args.data_dir,
        split="val",
        batch_size=1,
        num_workers=args.workers,
        image_size=args.image_size,
        roi_prob=0.0,
        roi_focus_prob=0.0,
        seed=args.seed,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    return train_loader, val_loader


def _move_instances_to_device(instances: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved = []
    for sample in instances:
        moved.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
        )
    return moved


def _gather_refine_targets(refined_outputs: list[dict[str, Any]], proposals: list[list[dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_masks = []
    target_masks = []
    labels = []
    for sample_output, sample_proposals in zip(refined_outputs, proposals):
        for instance_output, proposal in zip(sample_output["instances"], sample_proposals):
            if "gt_mask" not in proposal:
                continue
            pred_masks.append(instance_output["mask_logits"].unsqueeze(0))
            target_masks.append(proposal["gt_mask"].unsqueeze(0).float())
            labels.append(int(proposal["category_id"]))
    if not pred_masks:
        return (
            torch.zeros((0, 1, 1, 1), device=refined_outputs[0]["instances"][0]["mask_logits"].device) if refined_outputs and refined_outputs[0]["instances"] else torch.zeros((0, 1, 1, 1)),
            torch.zeros((0, 1, 1, 1)),
            torch.zeros((0,), dtype=torch.long),
        )
    return torch.stack(pred_masks, dim=0), torch.stack(target_masks, dim=0), torch.tensor(labels, dtype=torch.long, device=pred_masks[0].device)


def _build_matched_proposals(decoded: list[list[dict[str, Any]]], instances: list[dict[str, Any]], device: torch.device) -> list[list[dict[str, Any]]]:
    batch_proposals = []
    for sample_decoded, sample_instances in zip(decoded, instances):
        matches = greedy_match_boxes(sample_decoded, sample_instances, iou_threshold=0.3)
        sample_proposals = []
        for proposal_idx, gt_idx in matches:
            proposal = sample_decoded[proposal_idx]
            sample_proposals.append(
                {
                    "bbox": proposal["bbox"].to(device),
                    "category_id": int(proposal["category_id"]),
                    "score": float(proposal["score"]),
                    "center": proposal["center"].to(device),
                    "gt_mask": sample_instances["masks"][gt_idx].to(device).float(),
                }
            )
        batch_proposals.append(sample_proposals)
    return batch_proposals


def main() -> None:
    args = parse_args()
    device, _ = init_device(seed=args.seed, cpu=args.cpu, gpu=args.gpu, affinity=None)
    train_loader, val_loader = _prepare_loaders(args)
    model = build_instsam_model(args, device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    proposal_criterion = CenterNetProposalLoss(hole_positive_weight=args.hole_positive_weight).to(device)
    refine_criterion = InstanceRefineLoss().to(device)
    run_dir = resolve_instsam_run_dir(args)

    if args.phase == "oracle":
        logger.info("Oracle phase selected. Use eval_instsam.py --oracle for quantitative evaluation.")
        return

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and args.amp_dtype == "fp16")
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    best_metric = -1.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        proposal_recall_meter = []
        hole_recall_meter = []
        for step_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            instances_cpu = batch["instances"]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                features = model.forward_backbone(images)
                proposal_outputs = model.forward_proposals(features)
                proposal_targets = model.proposal_head.build_targets(
                    instances_cpu,
                    image_size=args.image_size,
                    hole_positive_weight=args.hole_positive_weight,
                )
                proposal_loss_dict = proposal_criterion(proposal_outputs, proposal_targets)
                loss = proposal_loss_dict["loss"]

                if args.phase in {"refine", "joint"}:
                    instances = _move_instances_to_device(instances_cpu, device)
                    decoded = model.decode_proposals(proposal_outputs, image_size=args.image_size)
                    use_gt = epoch < max(int(args.epochs * args.refine_gt_ratio), 1)
                    proposals = model.build_gt_proposals(instances) if use_gt else _build_matched_proposals(decoded, instances, device)
                    fallback_needed = any(len(sample) == 0 for sample in proposals)
                    if fallback_needed:
                        proposals = model.build_gt_proposals(instances)
                    refined = model.refine_instances(
                        features=features,
                        proposals=proposals,
                        processed_sizes=batch["processed_size"],
                        output_sizes=batch["processed_size"],
                        apply_roi_refiner=args.enable_roi_refiner,
                    )
                    pred_masks, target_masks, labels = _gather_refine_targets(refined, proposals)
                    if pred_masks.numel() > 0:
                        refine_loss_dict = refine_criterion(pred_masks, target_masks.to(pred_masks.device), labels)
                        if args.phase == "refine":
                            loss = refine_loss_dict["loss"]
                        else:
                            loss = proposal_loss_dict["loss"] + refine_loss_dict["loss"]

            should_track_metrics = args.train_metrics_interval > 0 and (step_idx % args.train_metrics_interval == 0)
            if should_track_metrics:
                decoded = model.decode_proposals(proposal_outputs, image_size=args.image_size)
                proposal_recall_meter.extend([proposal_recall_at_k(sample, target, topk=100) for sample, target in zip(decoded, instances_cpu)])
                hole_recall_meter.extend(
                    [
                        proposal_recall_at_k([item for item in sample if int(item["category_id"]) == 2], {"boxes": target["boxes"][target["labels"] == 2], "labels": target["labels"][target["labels"] == 2]}, topk=100)
                        if int((target["labels"] == 2).sum().item()) > 0
                        else 1.0
                        for sample, target in zip(decoded, instances_cpu)
                    ]
                )

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_proposal_recall = sum(proposal_recall_meter) / len(proposal_recall_meter) if proposal_recall_meter else -1.0
        avg_hole_recall = sum(hole_recall_meter) / len(hole_recall_meter) if hole_recall_meter else -1.0
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
            f"proposal_recall@100={avg_proposal_recall:.4f} | hole_recall={avg_hole_recall:.4f}"
        )
        score = avg_loss * -1.0 if avg_hole_recall < 0 else (avg_hole_recall if args.phase == "proposal" else avg_proposal_recall + avg_hole_recall)
        last_path = os.path.join(run_dir, "last.pth")
        save_checkpoint(last_path, model=model, optimizer=optimizer, epoch=epoch, extra={"phase": args.phase})
        if score > best_metric:
            best_metric = score
            save_checkpoint(os.path.join(run_dir, "best.pth"), model=model, optimizer=optimizer, epoch=epoch, extra={"phase": args.phase})


if __name__ == "__main__":
    main()
