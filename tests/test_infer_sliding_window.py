"""Sliding-window inference smoke tests for S11."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.inferencer import WireCRHQInstSAMInferencer


class _FakeWindowModel(nn.Module):
    def forward(self, images: torch.Tensor, *, processed_sizes, prompt_source="pred", **kwargs):
        batch_size = int(images.shape[0])
        fused_batches = []
        for batch_index in range(batch_size):
            image = images[batch_index]
            mask = image[0] > 0.5
            if mask.sum() == 0:
                fused_batches.append(
                    {
                        "labels": torch.zeros((0,), dtype=torch.long),
                        "instance_scores": torch.zeros((0,), dtype=torch.float32),
                        "refined_mask_logits": torch.zeros((0, 1, 16, 16), dtype=torch.float32),
                        "boxes_xyxy": torch.zeros((0, 4), dtype=torch.float32),
                    }
                )
                continue
            ys, xs = torch.where(mask)
            box = torch.tensor([[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]], dtype=torch.float32)
            logits = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(16, 16), mode="nearest") * 12.0 - 6.0
            fused_batches.append(
                {
                    "labels": torch.tensor([1], dtype=torch.long),
                    "instance_scores": torch.tensor([0.95], dtype=torch.float32),
                    "refined_mask_logits": logits,
                    "boxes_xyxy": box,
                }
            )
        return {
            "eval_dict": {
                "fused_batches": fused_batches,
            }
        }


def _build_image() -> Image.Image:
    canvas = torch.zeros((3, 96, 96), dtype=torch.float32)
    canvas[:, 30:66, 30:66] = 1.0
    array = (canvas.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def test_sliding_window_cross_window_fusion_and_exports(tmp_path: Path) -> None:
    image = _build_image()
    inferencer = WireCRHQInstSAMInferencer(
        model=_FakeWindowModel(),
        device="cpu",
        sliding_window=64,
        overlap=0.5,
        score_thresh_label=0.1,
        score_thresh_hole=0.1,
        mask_nms_iou_label=0.6,
        mask_nms_iou_hole=0.5,
    )

    prediction = inferencer.predict_image(image)
    assert len(prediction["window_instances"]) > 1
    assert len(prediction["fused_instances"]) == 1
    assert len(prediction["instances"]) == 1

    exported = inferencer.export_predictions(
        image=image,
        prediction=prediction,
        output_dir=tmp_path,
        stem="sample",
    )

    json_path = Path(exported["json_path"])
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["instances"]) == 1
    assert (tmp_path / "vis" / "sample.png").is_file()
    assert (tmp_path / "masks" / "sample_inst_000.png").is_file()
    assert (tmp_path / "rectified_crops" / "sample_inst_000.png").is_file()
