"""
Main training script for WireCR-SAM.
"""

import os
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.sam_dataset import get_dataset_class_names, get_sam_dataloader, validate_dataset_config
from utils import logger
from utils.experiment_io import build_experiment_summary, resolve_run_dir, write_json
from utils.init import init_device, init_sam_model
from utils.parser import args
from utils.scheduler import FakeLR, WarmUpCosineAnnealingLR
from utils.solver import SAMCriterion, SAMTester, SAMTrainer


def _build_dataloader(split, subset_ratio=1.0):
    return get_sam_dataloader(
        data_root=args.data_dir,
        split=split,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        image_size=args.image_size,
        subset_ratio=subset_ratio,
        subset_seed=args.subset_seed,
        num_classes=args.num_classes,
    )


def _validate_dataset_args():
    validate_dataset_config(args.dataset, args.num_classes)


def _log_results(title, results):
    logger.info("\n" + "=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    logger.info(f'Loss: {results.get("loss", 0.0):.4f}')
    logger.info(f'IoU: {results["iou"]:.4f}')
    logger.info(f'Dice: {results["dice"]:.4f}')
    logger.info(f'Precision: {results["precision"]:.4f}')
    logger.info(f'Recall: {results["recall"]:.4f}')
    logger.info(f'F1 Score: {results["f1"]:.4f}')
    logger.info(f'Boundary F1: {results["boundary_f1"]:.4f}')
    logger.info(f'clDice: {results["cldice"]:.4f}')
    if "wire_iou" in results:
        logger.info(f'Wire IoU: {results["wire_iou"]:.4f}')
    if "hole_iou" in results:
        logger.info(f'Interface-hole IoU: {results["hole_iou"]:.4f}')
        logger.info(f'Interface-hole Recall: {results["hole_recall"]:.4f}')
    if "foreground_iou" in results:
        logger.info(f'Foreground IoU: {results["foreground_iou"]:.4f}')
    logger.info("=" * 60 + "\n")


def _collect_checkpoint_paths(run_dir, pretrained=None):
    checkpoint_paths = {}
    best_path = os.path.join(run_dir, "best_iou.pth")
    last_path = os.path.join(run_dir, "last.pth")
    if os.path.isfile(best_path):
        checkpoint_paths["best_iou"] = os.path.abspath(best_path)
    if os.path.isfile(last_path):
        checkpoint_paths["last"] = os.path.abspath(last_path)
    if pretrained is not None and os.path.isfile(pretrained):
        checkpoint_paths["pretrained"] = os.path.abspath(pretrained)
    return checkpoint_paths


def _export_summary(*, run_dir, class_names, model, train_loader, val_loader, test_loader, results, best_iou, stage):
    summary = build_experiment_summary(
        args=args,
        run_dir=run_dir,
        class_names=class_names,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
        test_samples=len(test_loader.dataset),
        model=model,
        results=results,
        best_iou=best_iou,
        checkpoint_paths=_collect_checkpoint_paths(run_dir, pretrained=args.pretrained),
        stage=stage,
    )

    summary_path = args.results_json or os.path.join(run_dir, "experiment_summary.json")
    write_json(summary_path, summary)
    logger.info(f"=> Structured summary saved to: {summary_path}")


def main():
    logger.info("=" * 60)
    logger.info("WireCR-SAM - Training")
    logger.info("=" * 60)

    args.mode = "sam"
    if args.sam_checkpoint is None:
        logger.error("Error: --sam-checkpoint is required for SAM mode")
        sys.exit(1)
    _validate_dataset_args()

    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)
    class_names = get_dataset_class_names(args.dataset, args.num_classes)
    run_dir = resolve_run_dir(args)

    logger.info(f"=> Loading dataset: {args.dataset}")
    logger.info(f"=> Data directory: {args.data_dir}")
    logger.info(f"=> Subset ratio: {args.subset_ratio}")
    logger.info(f"=> Class names: {', '.join(class_names)}")
    logger.info(f"=> Run directory: {run_dir}")

    train_loader = _build_dataloader("train", subset_ratio=args.subset_ratio)
    val_loader = _build_dataloader("val")

    try:
        test_loader = _build_dataloader("test")
    except FileNotFoundError:
        logger.warning("Test split not found, using validation split for testing.")
        test_loader = val_loader

    logger.info(f"=> Train samples: {len(train_loader.dataset)}")
    logger.info(f"=> Val samples: {len(val_loader.dataset)}")
    logger.info(f"=> Test samples: {len(test_loader.dataset)}")

    logger.info("\n=> Initializing model...")
    model = init_sam_model(args)
    model.to(device)
    model.print_model_info()

    criterion = SAMCriterion(
        num_classes=args.num_classes,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_loss_weight,
        cldice_weight=args.cldice_weight,
        hole_class_weight=args.hole_class_weight,
    ).to(device)

    if args.evaluate:
        logger.info("\n=> Running evaluation only...")
        tester = SAMTester(model, device, criterion)
        results = tester(test_loader)
        _log_results("Evaluation Results", results)
        _export_summary(
            run_dir=run_dir,
            class_names=class_names,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            results=results,
            best_iou=results.get("iou"),
            stage="evaluate",
        )
        return

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    logger.info(f'=> Trainable parameters: {sum(param.numel() for param in trainable_params):,}')

    optimizer = optim.Adam(trainable_params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
    total_steps = (args.epochs if args.epochs else 100) * len(train_loader)
    warmup_steps = min(10 * len(train_loader), total_steps)

    if args.scheduler == "cosine":
        scheduler = WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            T_warmup=warmup_steps,
            eta_min=1e-6,
        )
    else:
        scheduler = FakeLR(optimizer=optimizer)

    subset_tag = "" if args.subset_ratio >= 1.0 else f"_subset{int(args.subset_ratio * 100):02d}"
    save_path = run_dir

    trainer = SAMTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        save_path=save_path,
        print_freq=20,
        val_freq=args.val_freq,
        test_freq=args.test_freq,
    )

    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info(f"=> Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            trainer.cur_epoch = checkpoint["epoch"] + 1
            trainer.best_iou = checkpoint.get("best_iou", 0.0)
            logger.info(f'=> Resumed from epoch {checkpoint["epoch"]}')
        else:
            logger.warning(f"=> Checkpoint not found: {args.resume}")

    epochs = args.epochs if args.epochs else 100
    logger.info("\n=> Training configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info("  Learning rate: 1e-4")
    logger.info(f"  Scheduler: {args.scheduler}")
    logger.info(f"  Validation frequency: every {args.val_freq} epoch(s)")
    logger.info(f"  Test frequency: every {args.test_freq} epoch(s)")
    logger.info(f"  Adapter size: {args.adapter_size}")
    logger.info(f"  Compression ratio: 1/{args.compression_ratio}")
    logger.info(f"  Class-aware prompts: {args.class_aware_prompts}")
    logger.info(f"  Freeze encoder: {args.freeze_encoder}")
    logger.info(f"  Freeze decoder: {args.freeze_decoder}")
    logger.info(f"  Boundary loss weight: {args.boundary_loss_weight}")
    logger.info(f"  clDice weight: {args.cldice_weight}")
    logger.info(f"  Hole class weight: {args.hole_class_weight}")
    logger.info("")

    logger.info("=> Starting training...\n")
    trainer.loop(epochs, train_loader, val_loader, test_loader)

    logger.info("\n=> Running final evaluation...")
    tester = SAMTester(model, device, criterion)
    results = tester(test_loader)
    _log_results("Final Test Results", results)
    logger.info(f'Best IoU achieved: {trainer.best_iou:.4f}')
    _export_summary(
        run_dir=run_dir,
        class_names=class_names,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        results=results,
        best_iou=trainer.best_iou,
        stage="train",
    )


if __name__ == "__main__":
    main()
