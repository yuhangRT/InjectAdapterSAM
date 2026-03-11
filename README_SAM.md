# CRNet-SAM Adapter Integration

## Overview

This project integrates CRNet as a feature enhancement adapter for the Segment Anything Model (SAM). The adapter processes SAM's ViT encoder output through a modified CRNet architecture to enhance features before passing them to SAM's mask decoder.

## Quick Start

### 1. Install Dependencies

```bash
# Install SAM dependencies
pip install -r requirements_sam.txt

# Initialize SAM submodule (if not already done)
git submodule update --init --recursive
```

### 2. Download SAM Checkpoint

Download the SAM checkpoint from [Facebook Research](https://github.com/facebookresearch/segment-anything#model-checkpoints):

```bash
# For ViT-H (default)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

### 3. Download Dataset

For COCO dataset:
```bash
# Download COCO 2017 dataset
# Update DATA_DIR in scripts/grid_search.sh with your path
```

### 4. Train SAM with CRNet Adapter

```bash
python main_sam.py \
    --mode sam \
    --sam-model-type vit_h \
    --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
    --dataset coco \
    --data-dir /path/to/coco \
    --adapter-size medium \
    --compression-ratio 8 \
    --epochs 100 \
    --batch-size 8 \
    --workers 4 \
    --prompt-strategy random \
    --num-prompts 1 \
    --gpu 0
```

## Architecture

```
SAM Image Encoder (ViT-H)
        ↓
  (batch, 256, 64, 64)
        ↓
┌─────────────────────────┐
│   CRNet Feature Adapter  │
│  - Encoder: Dual-path    │
│    multi-scale convs     │
│  - Optional FC Bottleneck│
│  - Decoder: CRBlocks     │
│  - Residual Connection   │
└─────────────────────────┘
        ↓
  Enhanced Features
  (batch, 256, 64, 64)
        ↓
SAM Mask Decoder + Prompts
        ↓
   Segmentation Output
```

## Adapter Configurations

| Adapter Size | Encoder Channels | Decoder Channels | CRBlock Channels | Approx Params |
|--------------|------------------|------------------|------------------|---------------|
| **small**    | 32               | 64               | 64               | ~2M           |
| **medium**   | 64               | 128              | 128              | ~5M           |
| **large**    | 128              | 256              | 256              | ~10M          |

## Compression Ratios

Supported compression ratios: 4, 8, 16, 32, 64

- **1/4**: Minimal compression, highest capacity
- **1/8**: Balanced performance
- **1/16**: Good trade-off
- **1/32, 1/64**: Higher compression, lower capacity

## Command-Line Arguments

### SAM-Specific Arguments

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--mode` | `csi` | `csi`, `sam` | Operation mode |
| `--sam-model-type` | `vit_h` | `vit_h`, `vit_l`, `vit_b` | SAM model variant |
| `--sam-checkpoint` | required | - | Path to SAM checkpoint |
| `--dataset` | `coco` | `coco`, `custom` | Dataset name |

### Adapter Configuration

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--adapter-size` | `medium` | `small`, `medium`, `large` | Adapter size variant |
| `--compression-ratio` | `4` | `4`, `8`, `16`, `32`, `64` | Compression ratio |
| `--use-residual` | `True` | - | Use residual connection |
| `--adapter-simple` | `False` | - | Use simplified adapter (no FC) |

### Training Options

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--freeze-encoder` | `True` | - | Freeze SAM encoder |
| `--freeze-decoder` | `False` | - | Freeze SAM decoder |
| `--prompt-strategy` | `random` | `random`, `center`, `grid`, `box` | Prompt generation |
| `--num-prompts` | `1` | - | Number of prompts per image |

## Grid Search

Run automated grid search over all configurations:

```bash
# Update paths in scripts/grid_search.sh first!
bash scripts/grid_search.sh
```

This will test:
- 3 adapter sizes × 5 compression ratios × 3 prompt strategies = **45 experiments**

### Aggregate Results

```bash
python scripts/aggregate_results.py \
    --log-dir ./logs/grid_search \
    --output ./logs/grid_search/results.csv \
    --plot-dir ./logs/grid_search/plots
```

## Project Structure

```
CRNet/
├── models/
│   ├── crnet.py                    # Original CRNet (CSI)
│   ├── crnet_blocks.py             # NEW: Reusable blocks
│   ├── crnet_adapter.py            # NEW: SAM adapter
│   └── sam_wrapper.py              # NEW: SAM integration
├── dataset/
│   ├── cost2100.py                 # Original CSI dataset
│   └── sam_dataset.py              # NEW: SAM dataset
├── utils/
│   ├── parser.py                   # EXTENDED: SAM arguments
│   ├── init.py                     # EXTENDED: SAM init
│   ├── solver.py                   # EXTENDED: SAMTrainer/SAMTester
│   └── sam_metrics.py              # NEW: Segmentation metrics
├── main.py                         # EXTENDED: Mode dispatch
├── main_sam.py                     # NEW: SAM training script
├── scripts/
│   ├── grid_search.sh              # NEW: Automated testing
│   └── aggregate_results.py        # NEW: Results aggregation
├── third_party/sam/                # SAM submodule
└── requirements_sam.txt            # SAM dependencies
```

## Evaluation Metrics

The following metrics are tracked:
- **IoU** (Intersection over Union): Primary metric
- **Dice coefficient**: Secondary metric
- **Precision, Recall, F1**: Additional segmentation metrics

## Example Workflows

### 1. Quick Test Run

```bash
python main_sam.py \
    --mode sam --sam-model-type vit_h \
    --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
    --dataset coco --data-dir /path/to/coco \
    --adapter-size medium --compression-ratio 8 \
    --epochs 10 --batch-size 4 --workers 2 --gpu 0
```

### 2. Evaluation Only

```bash
python main_sam.py \
    --mode sam --evaluate \
    --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
    --pretrained ./checkpoints/sam_coco_medium_cr08.pth \
    --dataset coco --data-dir /path/to/coco \
    --adapter-size medium --compression-ratio 8 \
    --batch-size 8 --gpu 0
```

### 3. Compare Adapter Sizes

```bash
# Small adapter
python main_sam.py ... --adapter-size small --compression-ratio 8

# Medium adapter
python main_sam.py ... --adapter-size medium --compression-ratio 8

# Large adapter
python main_sam.py ... --adapter-size large --compression-ratio 8
```

## Troubleshooting

### SAM Not Found

If you get "SAM not found" error:
```bash
git submodule update --init --recursive
```

Or install SAM via pip:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### CUDA Out of Memory

- Reduce `--batch-size`
- Use smaller SAM model: `--sam-model-type vit_b`
- Enable gradient checkpointing (modify code)

### Dataset Not Found

Ensure your COCO dataset is structured as:
```
/path/to/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Citation

If you use this code, please cite:

```bibtex
@article{crnet2019,
  title={Multi-resolution CSI Feedback with deep learning in Massive MIMO System},
  author={...},
  journal={...},
  year={2019}
}

@inproceedings{sam2023,
  title={Segment Anything},
  author={...},
  booktitle={ICCV},
  year={2023}
}
```

## License

This project is licensed under the same terms as the original CRNet and SAM projects.
