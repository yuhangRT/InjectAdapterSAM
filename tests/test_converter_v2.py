"""Tests for the S01 converter."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset.geometry_utils import bbox_xywh_from_mask, load_isat_json, multi_polygon_to_mask
from scripts.convert_isat_to_coco_v2 import export_coco, group_objects_to_instances, load_isat_sample


def test_image_sam_002_groups_to_expected_instances():
    annotation = load_isat_sample(Path("samDataset/image_sam_002.json"))
    instances = group_objects_to_instances(annotation["objects"], width=annotation["info"]["width"], height=annotation["info"]["height"])
    counts = {1: 0, 2: 0}
    for instance in instances:
        counts[instance["category_id"]] += 1
        assert len(instance["bbox"]) == 4
        assert len(instance["oriented_box"]) == 5
        assert len(instance["principal_axis"]) == 2
    assert counts == {1: 8, 2: 8}


def test_export_bbox_is_xywh(tmp_path: Path):
    dst_root = tmp_path / "coco_out"
    report = export_coco(
        src_root=Path("samDataset").resolve(),
        dst_root=dst_root,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        dedupe_threshold=6,
        split_mode="dedupe",
        min_instance_area=0.0,
        copy_images=False,
    )
    assert sum(report["images"].values()) > 0
    coco_json = None
    image_info = None
    first_ann = None
    for split in ("train", "val", "test"):
        annotation_path = dst_root / "annotations" / f"instances_{split}.json"
        candidate_json = load_isat_json(annotation_path)
        for candidate_image in candidate_json["images"]:
            if candidate_image["file_name"] == "image_sam_002.jpg":
                image_info = candidate_image
                first_ann = next(ann for ann in candidate_json["annotations"] if ann["image_id"] == image_info["id"])
                coco_json = candidate_json
                break
        if image_info is not None:
            break
    assert coco_json is not None and image_info is not None and first_ann is not None, "Expected image_sam_002 in one of the exported splits."
    bbox = first_ann["bbox"]
    assert len(bbox) == 4
    assert bbox[2] >= 0 and bbox[3] >= 0

    # Validate bbox semantics against the same mask source used by the exporter.
    source_annotation = load_isat_sample(Path("samDataset/image_sam_002.json"))
    grouped = group_objects_to_instances(
        source_annotation["objects"],
        width=source_annotation["info"]["width"],
        height=source_annotation["info"]["height"],
    )
    matched = next(
        instance
        for instance in grouped
        if instance["category_id"] == first_ann["category_id"] and instance["group_id"] == first_ann["group_id"]
    )
    mask = multi_polygon_to_mask(source_annotation["info"]["height"], source_annotation["info"]["width"], matched["polygons"])
    assert bbox == bbox_xywh_from_mask(mask)


def test_unknown_category_hard_fails():
    with pytest.raises(ValueError, match="Unsupported category"):
        group_objects_to_instances(
            [{"category": "hole", "group": 1, "segmentation": [0, 0, 1, 0, 1, 1, 0, 1]}],
            width=10,
            height=10,
        )
