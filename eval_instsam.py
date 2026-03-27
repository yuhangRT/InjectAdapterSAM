"""
Evaluation entrypoint for WireCR-InstSAM.
"""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Any

import torch

from dataset.wire_hole_instance_dataset import WireHoleInstanceDataset, get_instance_dataloader
from utils import logger
from utils.init import init_device
from utils.instsam_cli import add_common_instsam_args
from utils.instsam_runtime import build_instsam_model, filter_predictions, load_checkpoint_into_model, predictions_to_coco
from utils.instance_matcher import proposal_recall_at_k
from utils.instance_metrics import coco_eval_from_predictions, compute_industrial_instance_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WireCR-InstSAM.")
    add_common_instsam_args(parser)
    parser.add_argument("--checkpoint", default=None, help="Optional path to model checkpoint.")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--oracle", action="store_true", help="Use GT proposals instead of learned proposals.")
    parser.add_argument("--wire-score-thresh", type=float, default=None)
    parser.add_argument("--hole-score-thresh", type=float, default=None)
    parser.add_argument("--wire-mask-nms-iou", type=float, default=0.60)
    parser.add_argument("--hole-mask-nms-iou", type=float, default=0.60)
    parser.add_argument("--topk-final", type=int, default=50)
    return parser.parse_args()


def _move_instances_to_device(instances: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [{key: value.to(device) if torch.is_tensor(value) else value for key, value in sample.items()} for sample in instances]


def _scale_box_to_full(box: torch.Tensor, resize_scale: tuple[float, float], crop_box: list[float] | None) -> torch.Tensor:
    scale_y, scale_x = resize_scale
    scaled = box.detach().clone().float()
    scaled[[0, 2]] /= scale_x
    scaled[[1, 3]] /= scale_y
    if crop_box is not None:
        scaled[[0, 2]] += float(crop_box[0])
        scaled[[1, 3]] += float(crop_box[1])
    return scaled


def collect_predictions(args, model, data_loader, device: torch.device):
    model.eval()
    cached_predictions = []
    cached_targets = []
    cached_image_ids = []
    proposal_recalls = []
    hole_recalls = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            instances = _move_instances_to_device(batch["instances"], device)
            features = model.forward_backbone(images)
            proposal_outputs = model.forward_proposals(features)
            decoded = model.decode_proposals(proposal_outputs, image_size=args.image_size)
            proposal_recalls.extend([proposal_recall_at_k(sample, target, topk=100) for sample, target in zip(decoded, instances)])
            hole_recalls.extend(
                [
                    proposal_recall_at_k([item for item in sample if int(item["category_id"]) == 2], {"boxes": target["boxes"][target["labels"] == 2], "labels": target["labels"][target["labels"] == 2]}, topk=100)
                    if int((target["labels"] == 2).sum().item()) > 0
                    else 1.0
                    for sample, target in zip(decoded, instances)
                ]
            )

            proposals = model.build_gt_proposals(instances) if args.oracle else decoded
            refined = model.refine_instances(
                features=features,
                proposals=proposals,
                processed_sizes=batch["processed_size"],
                output_sizes=batch["full_image_size"],
                apply_roi_refiner=args.enable_roi_refiner,
            )

            for batch_idx, sample in enumerate(refined):
                processed_instances = []
                for proposal, instance_output in zip(proposals[batch_idx], sample["instances"]):
                    bbox_full = _scale_box_to_full(
                        proposal["bbox"],
                        resize_scale=batch["resize_scale"][batch_idx],
                        crop_box=batch["crop_box"][batch_idx],
                    )
                    processed_instances.append(
                        {
                            "instance_id": instance_output["instance_id"],
                            "category_id": instance_output["category_id"],
                            "score": float(instance_output["score"]),
                            "bbox_processed": proposal["bbox"].detach().clone().cpu(),
                            "bbox_full": bbox_full.cpu(),
                            "mask_processed": instance_output["mask_processed"].cpu(),
                            "mask_full": instance_output["mask"].cpu(),
                            "source_prompt": instance_output["source_prompt"],
                        }
                    )
                cached_predictions.append(processed_instances)
                cached_targets.append(batch["instances"][batch_idx])
                cached_image_ids.append(batch["image_id"][batch_idx])

    return cached_predictions, cached_targets, cached_image_ids, proposal_recalls, hole_recalls


def evaluate_threshold_grid(args, dataset, cached_predictions, cached_targets, image_ids):
    if args.wire_score_thresh is not None and args.hole_score_thresh is not None:
        grid = [(args.wire_score_thresh, args.hole_score_thresh)]
    else:
        grid = list(itertools.product((0.10, 0.15, 0.20, 0.25), repeat=2))

    best = None
    for wire_thresh, hole_thresh in grid:
        filtered = filter_predictions(
            cached_predictions,
            wire_threshold=wire_thresh,
            hole_threshold=hole_thresh,
            wire_nms_iou=args.wire_mask_nms_iou,
            hole_nms_iou=args.hole_mask_nms_iou,
            topk_per_class=args.topk_final,
        )
        coco_predictions = predictions_to_coco(batched_predictions=filtered, image_ids=image_ids)
        coco_metrics = coco_eval_from_predictions(dataset.coco, coco_predictions)
        industrial = compute_industrial_instance_metrics(batched_predictions=filtered, batched_targets=cached_targets)
        score_tuple = (coco_metrics["AP50"], industrial["hole_recall"])
        candidate = {
            "wire_thresh": wire_thresh,
            "hole_thresh": hole_thresh,
            "coco": coco_metrics,
            "industrial": industrial,
        }
        if best is None or score_tuple > (best["coco"]["AP50"], best["industrial"]["hole_recall"]):
            best = candidate
    return best


def main() -> None:
    args = parse_args()
    device, _ = init_device(seed=args.seed, cpu=args.cpu, gpu=args.gpu, affinity=None)
    data_loader = get_instance_dataloader(
        data_root=args.data_dir,
        split=args.split,
        batch_size=1,
        num_workers=args.workers,
        image_size=args.image_size,
        roi_prob=0.0,
        roi_focus_prob=0.0,
        seed=args.seed,
    )
    dataset = data_loader.dataset
    model = build_instsam_model(args, device)
    if args.checkpoint:
        load_checkpoint_into_model(model, args.checkpoint)
    else:
        logger.warning("No checkpoint provided. Evaluation will use the current initialized model weights.")
    cached_predictions, cached_targets, image_ids, proposal_recalls, hole_recalls = collect_predictions(args, model, data_loader, device)
    best = evaluate_threshold_grid(args, dataset, cached_predictions, cached_targets, image_ids)
    logger.info(f"Best thresholds: wire={best['wire_thresh']:.2f}, hole={best['hole_thresh']:.2f}")
    logger.info(f"COCO metrics: {best['coco']}")
    logger.info(f"Industrial metrics: {best['industrial']}")
    logger.info(
        f"Proposal recall@100={sum(proposal_recalls)/max(len(proposal_recalls), 1):.4f} | "
        f"Hole recall={sum(hole_recalls)/max(len(hole_recalls), 1):.4f}"
    )


if __name__ == "__main__":
    main()
