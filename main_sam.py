"""
Main training script for SAM with CRNet adapter.

This script trains and evaluates the Segment Anything Model (SAM) with
a CRNet feature enhancement adapter.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import logger
from utils.parser import parser, args
from utils.init import init_device, init_sam_model
from utils.solver import SAMTrainer, SAMTester, SAMCriterion
from utils.scheduler import WarmUpCosineAnnealingLR, FakeLR
from dataset.sam_dataset import get_sam_dataloader


def main():
    """Main training/evaluation function for SAM+CRNet adapter."""

    logger.info('=' * 60)
    logger.info('SAM with CRNet Adapter - Training')
    logger.info('=' * 60)

    # Override mode for SAM
    args.mode = 'sam'

    # Validate arguments
    if args.sam_checkpoint is None:
        logger.error("Error: --sam-checkpoint is required for SAM mode")
        logger.info("\nPlease download SAM checkpoints from:")
        logger.info("  ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        logger.info("  ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
        logger.info("  ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        sys.exit(1)

    # Initialize device
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create data loaders
    logger.info(f'=> Loading dataset: {args.dataset}')
    logger.info(f'=> Data directory: {args.data_dir}')

    train_loader = get_sam_dataloader(
        data_root=args.data_dir,
        split='train',
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prompt_strategy=args.prompt_strategy,
        num_prompts=args.num_prompts,
        image_size=1024,
    )

    val_loader = get_sam_dataloader(
        data_root=args.data_dir,
        split='val',
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prompt_strategy=args.prompt_strategy,
        num_prompts=args.num_prompts,
        image_size=1024,
    )

    test_loader = get_sam_dataloader(
        data_root=args.data_dir,
        split='val',  # Use val as test for now
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prompt_strategy=args.prompt_strategy,
        num_prompts=args.num_prompts,
        image_size=1024,
    )

    logger.info(f'=> Train samples: {len(train_loader.dataset)}')
    logger.info(f'=> Val samples: {len(val_loader.dataset)}')
    logger.info(f'=> Test samples: {len(test_loader.dataset)}')

    # Initialize model
    logger.info('\n=> Initializing model...')
    model = init_sam_model(args)
    model.to(device)

    # Print model info
    model.print_model_info()

    # Define loss function
    criterion = SAMCriterion(
        mask_weight=1.0,
        iou_weight=1.0,
        dice_weight=0.0,
    ).to(device)

    # Evaluation mode
    if args.evaluate:
        logger.info('\n=> Running evaluation only...')
        tester = SAMTester(model, device, criterion)
        results = tester(test_loader)
        logger.info('\n' + '=' * 60)
        logger.info('Evaluation Results')
        logger.info('=' * 60)
        logger.info(f'Mean IoU: {results["iou"]:.4f}')
        logger.info(f'Mean Dice: {results["dice"]:.4f}')
        logger.info(f'Precision: {results["precision"]:.4f}')
        logger.info(f'Recall: {results["recall"]:.4f}')
        logger.info(f'F1 Score: {results["f1"]:.4f}')
        logger.info('=' * 60 + '\n')
        return

    # Define optimizer (only optimize trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f'=> Trainable parameters: {sum(p.numel() for p in trainable_params):,}')

    optimizer = optim.Adam(
        trainable_params,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    # Define learning rate scheduler
    total_steps = args.epochs * len(train_loader) if args.epochs else 100 * len(train_loader)
    warmup_steps = 10 * len(train_loader)

    if args.scheduler == 'cosine':
        scheduler = WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            T_warmup=warmup_steps,
            eta_min=1e-6,
        )
    else:
        scheduler = FakeLR(optimizer=optimizer)

    # Create trainer
    save_path = os.path.join('./checkpoints',
                            f'sam_{args.dataset}_{args.adapter_size}_cr{args.compression_ratio}')

    trainer = SAMTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_prompts=args.num_prompts,
        prompt_strategy=args.prompt_strategy,
        save_path=save_path,
        print_freq=20,
        val_freq=10,
        test_freq=10,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info(f'=> Resuming from checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            trainer.cur_epoch = checkpoint['epoch'] + 1
            trainer.best_iou = checkpoint.get('best_iou', 0.0)
            logger.info(f'=> Resumed from epoch {checkpoint["epoch"]}')
        else:
            logger.warning(f'=> Checkpoint not found: {args.resume}')

    # Set number of epochs
    epochs = args.epochs if args.epochs else 100

    logger.info(f'\n=> Training configuration:')
    logger.info(f'  Epochs: {epochs}')
    logger.info(f'  Batch size: {args.batch_size}')
    logger.info(f'  Learning rate: 1e-4')
    logger.info(f'  Scheduler: {args.scheduler}')
    logger.info(f'  Adapter size: {args.adapter_size}')
    logger.info(f'  Compression ratio: 1/{args.compression_ratio}')
    logger.info(f'  Prompt strategy: {args.prompt_strategy}')
    logger.info(f'  Num prompts: {args.num_prompts}')
    logger.info(f'  Freeze encoder: {args.freeze_encoder}')
    logger.info(f'  Freeze decoder: {args.freeze_decoder}')
    logger.info('')

    # Start training
    logger.info('=> Starting training...\n')
    trainer.loop(epochs, train_loader, val_loader, test_loader)

    # Final evaluation
    logger.info('\n=> Running final evaluation...')
    tester = SAMTester(model, device, criterion)
    results = tester(test_loader)

    logger.info('\n' + '=' * 60)
    logger.info('Final Test Results')
    logger.info('=' * 60)
    logger.info(f'Mean IoU: {results["iou"]:.4f}')
    logger.info(f'Mean Dice: {results["dice"]:.4f}')
    logger.info(f'Precision: {results["precision"]:.4f}')
    logger.info(f'Recall: {results["recall"]:.4f}')
    logger.info(f'F1 Score: {results["f1"]:.4f}')
    logger.info(f'Best IoU achieved: {trainer.best_iou:.4f}')
    logger.info('=' * 60 + '\n')


if __name__ == '__main__':
    main()
