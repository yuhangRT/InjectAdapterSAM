from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from dataset.collate_v2 import collate_label_hole_batch
from dataset.label_hole_instance_dataset import LabelHoleInstanceDataset
from scripts.visualize_dataset_v2 import export_dataset_visualizations

DATA_ROOT = Path("samDataset_instance_coco")


def _make_dataset(**kwargs):
    default_kwargs = {
        "data_root": DATA_ROOT,
        "split": "train",
        "image_size": 256,
        "seed": 7,
        "augment": False,
    }
    default_kwargs.update(kwargs)
    return LabelHoleInstanceDataset(**default_kwargs)


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _assert_instance_alignment(instances, tolerance: float = 2.0) -> None:
    masks = instances["masks"]
    boxes = instances["boxes"].cpu().numpy()
    oriented_boxes = instances["oriented_boxes"].cpu().numpy()
    principal_axes = instances["principal_axes"].cpu().numpy()
    for index in range(int(masks.shape[0])):
        mask_bbox = _mask_bbox(masks[index].cpu().numpy())
        assert mask_bbox is not None
        assert np.allclose(boxes[index], mask_bbox, atol=tolerance)
        assert np.isfinite(oriented_boxes[index]).all()
        assert np.isfinite(principal_axes[index]).all()
        assert abs(float(np.linalg.norm(principal_axes[index])) - 1.0) < 1e-4


def _first_image_with_both_classes(dataset: LabelHoleInstanceDataset) -> int:
    for index, image_id in enumerate(dataset.image_ids):
        labels = {int(instance["label"]) for instance in dataset.load_original_instances(image_id)}
        if {1, 2}.issubset(labels):
            return index
    raise AssertionError("Could not find an image containing both label_sleeve and empty_terminal")


def test_full_image_smoke():
    dataset = _make_dataset(full_image_prob=1.0, object_crop_prob=0.0)
    sample = dataset[0]
    assert sample["crop_box"] is None
    assert sample["orig_size"][0] > 0 and sample["orig_size"][1] > 0
    assert sample["processed_size"] == (256, 256)
    assert tuple(sample["image"].shape) == (3, 256, 256)
    _assert_instance_alignment(sample["instances"])


def test_crop_smoke_hole_and_label():
    base_dataset = _make_dataset(full_image_prob=0.0, object_crop_prob=0.0)
    index = _first_image_with_both_classes(base_dataset)

    hole_dataset = _make_dataset(
        full_image_prob=0.0,
        object_crop_prob=0.0,
        hole_focused_prob=1.0,
        label_focused_prob=0.0,
    )
    hole_sample = hole_dataset[index]
    assert hole_sample["crop_box"] is not None
    assert 2 in hole_sample["instances"]["labels"].tolist()
    _assert_instance_alignment(hole_sample["instances"])

    label_dataset = _make_dataset(
        full_image_prob=0.0,
        object_crop_prob=0.0,
        hole_focused_prob=0.0,
        label_focused_prob=1.0,
    )
    label_sample = label_dataset[index]
    assert label_sample["crop_box"] is not None
    assert 1 in label_sample["instances"]["labels"].tolist()
    _assert_instance_alignment(label_sample["instances"])


def test_object_crop_smoke():
    dataset = _make_dataset(full_image_prob=0.0, object_crop_prob=1.0)
    sample = dataset[0]
    assert sample["crop_box"] is not None
    assert sample["crop_mode"] == "object"
    assert sample["instances"]["masks"].shape[0] >= 1
    _assert_instance_alignment(sample["instances"])


def test_batch_collate_smoke():
    dataset = _make_dataset(full_image_prob=1.0, object_crop_prob=0.0)
    batch2 = collate_label_hole_batch([dataset[i] for i in range(2)])
    assert tuple(batch2["image"].shape) == (2, 3, 256, 256)
    assert len(batch2["instances"]) == 2
    assert len(batch2["image_id"]) == 2

    batch4 = collate_label_hole_batch([dataset[i] for i in range(4)])
    assert tuple(batch4["image"].shape) == (4, 3, 256, 256)
    assert len(batch4["instances"]) == 4


def test_visualization_export(tmp_path):
    dataset = _make_dataset(full_image_prob=0.0, object_crop_prob=1.0)
    exported = export_dataset_visualizations(dataset, tmp_path, limit=2)
    assert len(exported) == 2
    for raw_path, crop_path in exported:
        assert raw_path.is_file()
        assert crop_path.is_file()
        assert raw_path.stat().st_size > 0
        assert crop_path.stat().st_size > 0


def test_visualization_cli_smoke(tmp_path):
    output_dir = tmp_path / "cli_vis"
    cmd = [
        sys.executable,
        "scripts/visualize_dataset_v2.py",
        "--data-root",
        str(DATA_ROOT),
        "--split",
        "train",
        "--output-dir",
        str(output_dir),
        "--limit",
        "1",
        "--image-size",
        "256",
        "--no-augment",
    ]
    subprocess.run(cmd, check=True, cwd=Path.cwd())
    raw_files = sorted((output_dir / "raw_overlay").glob("*.png"))
    crop_files = sorted((output_dir / "crop_overlay").glob("*.png"))
    assert len(raw_files) == 1
    assert len(crop_files) == 1
    assert raw_files[0].stat().st_size > 0
    assert crop_files[0].stat().st_size > 0
