#!/usr/bin/env sh
set -eu

# Prefer the environment's CUDA-enabled torch over any user-site override.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <coco_root> <output_dir> [config_path]" >&2
  exit 1
fi

COCO_ROOT="$1"
OUTPUT_DIR="$2"
CONFIG_PATH="${3:-configs/wirecr_hqinstsam_vitb.yaml}"
SUBSET_ROOT="${OUTPUT_DIR}/overfit8_data"
RUN_DIR="${OUTPUT_DIR}/overfit8_run"
RESUME_ARG=""
FORCE_FRESH="${OVERFIT_FORCE_FRESH:-1}"

mkdir -p "${SUBSET_ROOT}/annotations" "${SUBSET_ROOT}/images/train" "${SUBSET_ROOT}/images/val"

"${PYTHON_BIN}" - <<'PY' "${COCO_ROOT}" "${SUBSET_ROOT}"
import json
import shutil
import sys
from pathlib import Path

coco_root = Path(sys.argv[1]).expanduser().resolve()
subset_root = Path(sys.argv[2]).expanduser().resolve()
ann_path = coco_root / "annotations" / "instances_train.json"
if not ann_path.is_file():
    raise SystemExit(f"Missing annotation file: {ann_path}")

with ann_path.open("r", encoding="utf-8") as handle:
    coco = json.load(handle)

images = sorted(coco["images"], key=lambda item: int(item["id"]))[:8]
image_ids = {int(item["id"]) for item in images}
annotations = [ann for ann in coco["annotations"] if int(ann["image_id"]) in image_ids]
subset = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "categories": coco.get("categories", []),
    "images": images,
    "annotations": annotations,
}

dst_ann = subset_root / "annotations" / "instances_train.json"
with dst_ann.open("w", encoding="utf-8") as handle:
    json.dump(subset, handle, ensure_ascii=False, indent=2)

for image in images:
    file_name = image["file_name"]
    src = coco_root / "images" / "train" / file_name
    if not src.is_file():
        src = coco_root / "images" / "val" / file_name
    if not src.is_file():
        raise SystemExit(f"Missing source image: {file_name}")
    dst = subset_root / "images" / "train" / file_name
    shutil.copy2(src, dst)
    shutil.copy2(src, subset_root / "images" / "val" / file_name)

shutil.copy2(dst_ann, subset_root / "annotations" / "instances_val.json")
PY

if [ "${FORCE_FRESH}" != "1" ] && [ -f "${RUN_DIR}/last.pth" ]; then
  RESUME_ARG="--set runtime.resume=${RUN_DIR}/last.pth"
fi

"${PYTHON_BIN}" train_wirecr_hqinstsam.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${RUN_DIR}" \
  ${RESUME_ARG} \
  --set "data.root=${SUBSET_ROOT}" \
  --set "data.train_split=train" \
  --set "data.val_split=val" \
  --set "data.full_image_prob=0.5" \
  --set "data.object_crop_prob=0.5" \
  --set "data.hole_focused_prob=0.6" \
  --set "data.label_focused_prob=0.4" \
  --set "train.epochs=90" \
  --set "train.warmup_epochs=8" \
  --set "train.batch_size=1" \
  --set "train.val_batch_size=1" \
  --set "train.grad_accum_steps=1" \
  --set "train.amp=false" \
  --set "train.lr_lora=1e-5" \
  --set "train.lr_new_modules=5e-5" \
  --set "train.grad_clip_norm=0.05" \
  --set "train.workers=4" \
  --set "train.val_workers=2" \
  --set "train.pin_memory=true" \
  --set "train.persistent_workers=true" \
  --set "train.prefetch_factor=2" \
  --set "train.warmup_iters=32" \
  --set "train.val_interval=10" \
  --set "eval.score_grid_label=[0.2,0.35,0.5,0.65,0.8]" \
  --set "eval.score_grid_hole=[0.2,0.35,0.5,0.65,0.8]" \
  --set "eval.mask_prob_grid=[0.4,0.5,0.6]" \
  --set "eval.mask_nms_iou_label_grid=[0.3,0.4,0.5,0.6]" \
  --set "eval.mask_nms_iou_hole_grid=[0.25,0.35,0.45,0.55]"
