# InjectAdapterSAM

`WireCRHQInstSAM` is now the primary instance-segmentation mainline in this repository.

The maintained path is:

- training: `train_wirecr_hqinstsam.py`
- evaluation: `eval_wirecr_hqinstsam.py`
- inference: `infer_wirecr_hqinstsam.py`
- config: `configs/wirecr_hqinstsam_vitb.yaml`

Legacy modules are preserved only for history and must not be imported by the new mainline:

- `train_instsam.py`
- `eval_instsam.py`
- `infer_instsam.py`
- `models/sam_wrapper.py`
- `models/sam_fpn_segmentor.py`
- `models/wirecr_instsam.py`

## Task Definition

The new mainline only maintains two instance classes:

- `label_sleeve` (`wire`)
- `empty_terminal` (`interface-hole`)

Each prediction produces:

- `boxes`
- `labels`
- `scores`
- `masks`
- `prompt_meta`
- `rectified_crop` for `label_sleeve` exports

## Data Preparation

Convert ISAT annotations to the new COCO v2 format:

```bash
python3 scripts/convert_isat_to_coco_v2.py \
  --src ./samDataset \
  --dst ./samDataset_instance_coco \
  --split-mode dedupe \
  --dedupe-threshold 6
```

Visualize dataset samples:

```bash
python3 scripts/visualize_dataset_v2.py \
  --data-root ./samDataset_instance_coco \
  --split train \
  --output-dir ./tmp_vis_v2 \
  --limit 8
```

## Training

Standard training:

```bash
python3 train_wirecr_hqinstsam.py \
  --config configs/wirecr_hqinstsam_vitb.yaml \
  --output-dir ./runs/wirecr_hqinstsam_vitb
```

Useful overrides:

```bash
python3 train_wirecr_hqinstsam.py \
  --config configs/wirecr_hqinstsam_vitb.yaml \
  --output-dir ./runs/wirecr_hqinstsam_debug \
  --set train.epochs=2 \
  --set train.batch_size=1 \
  --set train.val_batch_size=1 \
  --set train.workers=0
```

Resume training:

```bash
python3 train_wirecr_hqinstsam.py \
  --config configs/wirecr_hqinstsam_vitb.yaml \
  --resume ./runs/wirecr_hqinstsam_vitb/last.pth
```

Checkpoints written by the unified trainer:

- `last.pth`
- `best_ap.pth`
- `best_ap50.pth`
- `best_hole_recall.pth`

Checkpoint metadata includes:

- `best_val_thresholds`
- `val_metrics_summary`
- `config_snapshot`

## Evaluation

Run validation with threshold search:

```bash
python3 eval_wirecr_hqinstsam.py \
  --config configs/wirecr_hqinstsam_vitb.yaml \
  --checkpoint ./runs/wirecr_hqinstsam_vitb/best_ap.pth \
  --output ./runs/wirecr_hqinstsam_vitb/eval_metrics.json
```

Export the thresholds stored in a checkpoint:

```bash
python3 scripts/export_best_thresholds.py \
  --checkpoint ./runs/wirecr_hqinstsam_vitb/best_ap.pth \
  --output ./runs/wirecr_hqinstsam_vitb/best_thresholds.json
```

Metrics include:

- `mask_ap`
- `AP50`
- `AP75`
- `per_class_AP50`
- `empty_terminal_recall`
- `label_sleeve_boundary_f1`
- `mean_mask_iou`
- `count_mae`
- `merge_error_count`
- `split_error_count`

## Inference

Single image:

```bash
python3 infer_wirecr_hqinstsam.py \
  --config configs/wirecr_hqinstsam_vitb.yaml \
  --checkpoint ./runs/wirecr_hqinstsam_vitb/best_ap.pth \
  --image ./demo/example.png \
  --output-dir ./outputs/example
```

Directory inference:

```bash
python3 infer_wirecr_hqinstsam.py \
  --config configs/wirecr_hqinstsam_vitb.yaml \
  --checkpoint ./runs/wirecr_hqinstsam_vitb/best_ap.pth \
  --image-dir ./demo/images \
  --output-dir ./outputs/batch
```

Inference defaults:

- sliding window: `1024`
- overlap: `0.2`
- thresholds: checkpoint `best_val_thresholds` first, YAML fallback second

Exports:

- `json/`
- `vis/`
- `masks/`
- `rectified_crops/`

## Overfit8

Prepare and run the fixed 8-image overfit subset:

```bash
sh scripts/run_overfit8.sh ./samDataset_instance_coco ./runs/overfit8
```

The script creates an `overfit8_data` subset and launches the unified trainer on it.

## Tests

Core regression suite:

```bash
pytest tests/test_backbone_v2.py \
  tests/test_pixel_decoder.py \
  tests/test_query_head.py \
  tests/test_prompt_builder_v2.py \
  tests/test_hq_refiner.py \
  tests/test_end2end_smoke.py \
  tests/test_trainer_smoke.py \
  tests/test_evaluator_metrics.py \
  tests/test_infer_sliding_window.py -q
```

## Known Limits

- `vit_b` is the only validated training path in the current mainline.
- `vit_l` remains a compatibility template and has not passed the full regression path.
- The current evaluator uses repository-local industrial metrics plus COCO mask evaluation when a COCO ground-truth handle is available.
- Overfit8 acceptance still depends on actual training runtime and dataset availability.
