# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRNet is a PyTorch implementation for CSI (Channel State Information) feedback in Massive MIMO systems, based on the paper [Multi-resolution CSI Feedback with deep learning in Massive MIMO System](https://arxiv.org/abs/1910.14322). The project implements an encoder-decoder neural network for compressing and reconstructing wireless channel state information.

## Common Commands

### Training from Scratch
```bash
python main.py \
  --data-dir '/path/to/COST2100' \
  --scenario 'in' \
  --epochs 2500 \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --scheduler cosine \
  --gpu 0
```

### Evaluation with Pre-trained Model
```bash
python main.py \
  --data-dir '/path/to/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/in_04' \
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --cpu
```

### Resume Training from Checkpoint
Add `--resume './checkpoints/last.pth'` to the training command.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--data-dir` | Path to COST2100 dataset directory (required) |
| `--scenario` | Channel scenario: `in` (indoor) or `out` (outdoor) (required) |
| `--cr` | Compression ratio reciprocal (4, 8, 16, 32, or 64) |
| `--scheduler` | Learning rate scheduler: `const` or `cosine` |
| `--evaluate` | Run evaluation mode instead of training |
| `--pretrained` | Path to pre-trained checkpoint for evaluation |
| `--resume` | Path to checkpoint for resuming training |

## Code Architecture

### Entry Point
- **[main.py](main.py)** - Main script that initializes environment, data loaders, model, and training loop

### Model Architecture
- **[models/crnet.py](models/crnet.py)** - CRNet model implementation
  - `ConvBN`: Conv2d + BatchNorm2d building block
  - `CRBlock`: Residual block with dual-path architecture (kernel sizes: 3x3, 1x9, 9x1, 1x5, 5x1)
  - `CRNet`: Encoder-decoder architecture
    - Encoder: Two parallel convolutional paths → concat → 1x1 conv → FC compression
    - Decoder: FC decompression → 5x5 conv → 2x CRBlock → Sigmoid
  - Input shape: `(batch, 2, 32, 32)` - 2 channels for real/imaginary parts of CSI matrix
  - Compression controlled by `reduction` parameter (compression ratio = 1/reduction)

### Data Loading
- **[dataset/cost2100.py](dataset/cost2100.py)** - COST2100 dataset loader
  - `Cost2100DataLoader`: Creates train/val/test loaders from .mat files
  - `PreFetcher`: GPU data pre-fetcher for accelerated training (CUDA streams)
  - Expects files: `DATA_Htrain{scenario}.mat`, `DATA_Hval{scenario}.mat`, `DATA_Htest{scenario}.mat`, `DATA_HtestF{scenario}_all.mat`

### Training Infrastructure
- **[utils/solver.py](utils/solver.py)** - Training and testing pipelines
  - `Trainer`: Main training loop with validation/testing at intervals
    - Saves checkpoints: `last.pth`, `best_rho.pth`, `best_nmse.pth`
    - State includes: epoch, model, optimizer, scheduler, best_rho, best_nmse
  - `Tester`: Evaluation loop computing loss, rho (correlation), NMSE

- **[utils/scheduler.py](utils/scheduler.py)** - Learning rate schedulers
  - `WarmUpCosineAnnealingLR`: Cosine annealing with warmup phase
  - `FakeLR`: Constant learning rate placeholder

- **[utils/init.py](utils/init.py)** - Initialization utilities
  - `init_device`: Sets up CPU/GPU, random seed, CPU affinity
  - `init_model`: Creates CRNet, loads pretrained weights, counts FLOPs via thop

- **[utils/statics.py](utils/statics.py)** - Metrics and utilities
  - `AverageMeter`: Running average of values
  - `evaluator`: Computes NMSE (normalized mean square error) and rho (correlation coefficient)

- **[utils/parser.py](utils/parser.py)** - CLI argument definitions

### Model Metrics
- **NMSE** (Normalized Mean Square Error): Lower is better (negative dB values)
- **Rho** (Correlation Coefficient): Higher is better (closer to 1)
- Training uses MSE loss; evaluation uses both NMSE and rho

## Project Structure Expectations

```
home/
├── CRNet/              # This repository
│   ├── dataset/
│   ├── models/
│   ├── utils/
│   └── main.py
├── COST2100/           # Dataset directory
│   ├── DATA_Htrainin.mat
│   ├── DATA_Hvalin.mat
│   ├── DATA_Htestin.mat
│   └── DATA_HtestFin_all.mat
└── Experiments/
    └── checkpoints/    # Model checkpoints saved here
```

## Notes

- The model uses LeakyReLU(negative_slope=0.3) throughout
- Xavier uniform initialization is used for conv and linear layers
- FLOPs are computed using thop library (exclude BatchNorm in reported values)
- Cosine scheduler with warmup is recommended for best results (2500 epochs, lr_init=2e-3)
- Compression ratio is specified via `--cr` as the reciprocal (e.g., `--cr 4` means 1/4 compression)
