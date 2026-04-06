"""Export best validation thresholds from a WireCR-HQInstSAM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export best validation thresholds from checkpoint metadata.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--output", required=True, help="Path to output JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu")
    payload = {
        "best_val_thresholds": checkpoint.get("best_val_thresholds", {}),
        "best_metrics": checkpoint.get("best_metrics", {}),
        "val_metrics_summary": checkpoint.get("val_metrics_summary", {}),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
