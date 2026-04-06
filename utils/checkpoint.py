"""Checkpoint helpers for the WireCR-HQInstSAM training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import amp as torch_amp

__all__ = [
    "DEFAULT_CHECKPOINT_FILENAMES",
    "build_checkpoint_state",
    "save_checkpoint",
    "load_checkpoint",
]


DEFAULT_CHECKPOINT_FILENAMES = {
    "last": "last.pth",
    "best_ap": "best_ap.pth",
    "best_ap50": "best_ap50.pth",
    "best_hole_recall": "best_hole_recall.pth",
}


def build_checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch_amp.GradScaler | None = None,
    epoch: int,
    global_step: int,
    best_metrics: Mapping[str, float] | None = None,
    val_metrics_summary: Mapping[str, Any] | None = None,
    best_val_thresholds: Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "best_metrics": dict(best_metrics or {}),
        "val_metrics_summary": dict(val_metrics_summary or {}),
        "best_val_thresholds": dict(best_val_thresholds or {}),
        "config_snapshot": dict(config_snapshot or {}),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra_state:
        state["extra_state"] = dict(extra_state)
    return state


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(dict(state), temp_path)
    temp_path.replace(checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch_amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state and state["scaler"]:
        scaler.load_state_dict(state["scaler"])
    return state
