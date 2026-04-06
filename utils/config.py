"""YAML-first config helpers for WireCR-HQInstSAM S01."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "build_config_parser",
    "load_yaml_config",
    "merge_config",
    "load_config_from_args",
    "save_config",
]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected YAML mapping at {config_path}, got {type(loaded).__name__}")
    return loaded


def _coerce_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        next_node = cursor.get(key)
        if not isinstance(next_node, dict):
            next_node = {}
            cursor[key] = next_node
        cursor = next_node
    cursor[parts[-1]] = value


def merge_config(base: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if "." in key:
            _set_nested(merged, key, value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config_parser(description: str = "WireCR-HQInstSAM config loader") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--output-dir", type=str, default=None, help="Run directory override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index override.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted keys, e.g. train.epochs=60.",
    )
    return parser


def _parse_override_items(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = _coerce_value(raw_value.strip())
    return overrides


def load_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    base = load_yaml_config(args.config)
    overrides = _parse_override_items(getattr(args, "overrides", []))
    if getattr(args, "resume", None) is not None:
        overrides["runtime.resume"] = args.resume
    if getattr(args, "output_dir", None) is not None:
        overrides["runtime.output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        overrides["runtime.seed"] = args.seed
    if getattr(args, "gpu", None) is not None:
        overrides["runtime.gpu"] = args.gpu
    return merge_config(base, overrides)


def save_config(config: dict[str, Any], run_dir: str | Path) -> Path:
    output_dir = Path(run_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    json_path = output_dir / "config.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return path
