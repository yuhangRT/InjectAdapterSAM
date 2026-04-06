# WireCR-HQInstSAM Migration Guide

## New Mainline

Use these entrypoints for all new work:

- training: `train_wirecr_hqinstsam.py`
- evaluation: `eval_wirecr_hqinstsam.py`
- inference: `infer_wirecr_hqinstsam.py`
- config: `configs/wirecr_hqinstsam_vitb.yaml`

## Supported Task

The maintained task is two-class instance segmentation:

- `label_sleeve`
- `empty_terminal`

The old mixed semantic/instance routing is no longer the primary path.

## Old To New Mapping

- `sam_wrapper.py` -> `models/sam_backbone_v2.py` + `models/wirecr_hq_instsam.py`
- `wirecr_instsam.py` -> `models/wirecr_hq_instsam.py`
- legacy prompt heuristics -> `models/prompt_builder_v2.py`
- legacy SAM decoder usage -> `models/hq_mask_decoder.py`
- legacy scattered train flows -> `engine/trainer.py` + `train_wirecr_hqinstsam.py`
- legacy ad-hoc eval scripts -> `engine/evaluator.py` + `eval_wirecr_hqinstsam.py`
- legacy ad-hoc inference scripts -> `engine/inferencer.py` + `infer_wirecr_hqinstsam.py`

## Deprecated Modules

The new mainline must not import:

- `train_instsam.py`
- `eval_instsam.py`
- `infer_instsam.py`
- `models/sam_wrapper.py`
- `models/sam_fpn_segmentor.py`
- `models/wirecr_instsam.py`

These files remain only for historical comparison.

## Data Path Migration

Old flat ISAT annotations are no longer consumed directly by the training entry.

Required path:

1. Convert to COCO v2 with `scripts/convert_isat_to_coco_v2.py`
2. Train/evaluate/infer against the converted root

Example:

```bash
python3 scripts/convert_isat_to_coco_v2.py \
  --src ./samDataset \
  --dst ./samDataset_instance_coco \
  --split-mode dedupe
```

## Runtime Behavior Changes

- trainer is single-path only; there is no proposal-only / refine-only / oracle branch
- validation now produces threshold metadata for inference reuse
- inference now prefers checkpoint thresholds over hard-coded defaults
- sliding-window inference always does cross-window fusion before final class-wise mask NMS

## When Not To Use The Old Mainline

Do not use the old mainline when:

- you need the current two-class instance benchmark path
- you need threshold search and checkpoint metadata
- you need sliding-window fused inference exports
- you need the checklist-reviewed code path

The old path is only acceptable for historical reproduction.
