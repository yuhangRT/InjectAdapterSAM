"""Smoke tests for the S09 unified trainer and checkpoint flow."""

from __future__ import annotations

from pathlib import Path

import torch

from engine.trainer import WireCRHQInstSAMTrainer
from models.wirecr_hq_instsam import WireCRHQInstSAM
from utils.checkpoint import load_checkpoint


def _build_test_model(*, num_queries: int = 4) -> WireCRHQInstSAM:
    return WireCRHQInstSAM(
        model_type="vit_b",
        image_size=64,
        prompt_embed_dim=32,
        feature_dim=32,
        encoder_embed_dim=64,
        encoder_depth=12,
        encoder_num_heads=4,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        wirecr_out_channels=256,
        pixel_decoder_channels=256,
        num_queries=num_queries,
        query_decoder_layers=2,
    )


def _sample_instances() -> dict[str, torch.Tensor]:
    masks = torch.zeros((2, 64, 64), dtype=torch.uint8)
    masks[0, 12:28, 8:48] = 1
    masks[1, 20:44, 20:44] = 1
    return {
        "boxes": torch.tensor(
            [
                [8.0, 12.0, 48.0, 28.0],
                [20.0, 20.0, 44.0, 44.0],
            ],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "masks": masks,
        "areas": masks.flatten(1).sum(dim=1).float(),
        "iscrowd": torch.zeros((2,), dtype=torch.long),
        "group_ids": torch.tensor([0, 1], dtype=torch.long),
        "oriented_boxes": torch.tensor(
            [
                [28.0, 20.0, 40.0, 16.0, 0.0],
                [32.0, 32.0, 24.0, 24.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "principal_axes": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    }


def _sample_batch(seed: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "image": torch.rand((1, 3, 32, 40), generator=generator),
        "image_id": [seed],
        "image_path": [f"sample_{seed}.png"],
        "orig_size": [(32, 40)],
        "processed_size": [(64, 64)],
        "crop_box": [None],
        "crop_mode": ["full"],
        "instances": [_sample_instances()],
    }


def test_trainer_runs_one_epoch_and_writes_named_checkpoints(tmp_path: Path) -> None:
    model = _build_test_model()
    train_loader = [_sample_batch(0), _sample_batch(1)]
    val_loader = [_sample_batch(2)]
    trainer = WireCRHQInstSAMTrainer(
        model=model,
        output_dir=tmp_path,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        epochs=2,
        warmup_epochs=1,
        grad_accum_steps=2,
        amp=True,
        config_snapshot={"train": {"epochs": 2}},
        infer_config={"score_thresh_label": "auto_from_val", "score_thresh_hole": "auto_from_val"},
    )

    base_lrs = {
        group["group_name"]: trainer.scheduler.base_lrs[index]
        for index, group in enumerate(trainer.optimizer.param_groups)
    }
    assert base_lrs["lora"] == 5e-5
    assert base_lrs["wirecr_adapter"] == 2e-4
    assert base_lrs["pixel_decoder"] == 2e-4

    history = trainer.fit()

    assert len(history) == 2
    assert history[0]["prompt_source"] == "gt"
    assert history[0]["prompt_gt_ratio"] == 1.0
    assert history[0]["refine_loss_boost"] == 1.5
    assert history[1]["prompt_source"] == "mixed"
    assert history[1]["prompt_gt_ratio"] == 0.7
    assert "mask_ap" in history[1]
    assert "empty_terminal_recall" in history[1]

    for checkpoint_name in ("last.pth", "best_ap.pth", "best_ap50.pth", "best_hole_recall.pth"):
        assert (tmp_path / checkpoint_name).is_file()

    checkpoint = torch.load(tmp_path / "last.pth", map_location="cpu")
    assert "config_snapshot" in checkpoint
    assert checkpoint["config_snapshot"]["train"]["epochs"] == 2
    assert "best_val_thresholds" in checkpoint
    assert checkpoint["best_val_thresholds"]["score_thresh_label"] == 0.5
    assert checkpoint["best_val_thresholds"]["score_thresh_hole"] == 0.5
    assert "val_metrics_summary" in checkpoint
    assert checkpoint["epoch"] == 2


def test_trainer_resume_restores_epoch_optimizer_and_scheduler(tmp_path: Path) -> None:
    first_model = _build_test_model()
    train_loader = [_sample_batch(3), _sample_batch(4)]
    trainer = WireCRHQInstSAMTrainer(
        model=first_model,
        output_dir=tmp_path,
        train_loader=train_loader,
        val_loader=[_sample_batch(5)],
        device="cpu",
        epochs=1,
        warmup_epochs=1,
        grad_accum_steps=1,
        amp=False,
        config_snapshot={"runtime": {"resume": None}},
    )
    trainer.fit()
    checkpoint_path = tmp_path / "last.pth"
    assert checkpoint_path.is_file()

    resumed_model = _build_test_model()
    resumed_trainer = WireCRHQInstSAMTrainer(
        model=resumed_model,
        output_dir=tmp_path,
        train_loader=train_loader,
        val_loader=[_sample_batch(6)],
        device="cpu",
        epochs=2,
        warmup_epochs=1,
        grad_accum_steps=1,
        amp=False,
        resume_path=checkpoint_path,
    )

    assert resumed_trainer.state.epoch == 1
    assert resumed_trainer.state.global_step > 0

    history = resumed_trainer.fit()
    assert len(history) == 1

    reloaded_model = _build_test_model()
    state = load_checkpoint(checkpoint_path, model=reloaded_model)
    assert state["epoch"] == 2


def test_trainer_to_device_preserves_processed_size_for_matcher() -> None:
    trainer = WireCRHQInstSAMTrainer(
        model=_build_test_model(),
        output_dir=Path("/tmp/wirecr_hqinstsam_test"),
        train_loader=[_sample_batch(7)],
        val_loader=[_sample_batch(8)],
        device="cpu",
        epochs=1,
        warmup_epochs=1,
        grad_accum_steps=1,
        amp=False,
    )

    batch = _sample_batch(9)
    moved = trainer._to_device(batch["instances"], torch.device("cpu"), processed_sizes=batch["processed_size"])

    assert moved[0]["processed_size"] == (64, 64)


def test_load_checkpoint_ignores_legacy_score_fusion_mlp_weights(tmp_path: Path) -> None:
    model = _build_test_model()
    checkpoint_path = tmp_path / "legacy_score_fusion.pth"
    state = {
        "model": {
            **model.state_dict(),
            "score_fusion.mlp.0.weight": torch.randn(8, 4),
            "score_fusion.mlp.0.bias": torch.randn(8),
            "score_fusion.mlp.2.weight": torch.randn(1, 8),
            "score_fusion.mlp.2.bias": torch.randn(1),
        }
    }
    torch.save(state, checkpoint_path)

    reloaded_model = _build_test_model()
    loaded = load_checkpoint(checkpoint_path, model=reloaded_model)

    assert "model" in loaded


def test_load_checkpoint_ignores_missing_score_fusion_weights_from_parameter_free_version(tmp_path: Path) -> None:
    model = _build_test_model()
    checkpoint_path = tmp_path / "parameter_free_score_fusion.pth"
    state = {
        "model": {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("score_fusion.")
        }
    }
    torch.save(state, checkpoint_path)

    reloaded_model = _build_test_model()
    loaded = load_checkpoint(checkpoint_path, model=reloaded_model)

    assert "model" in loaded
