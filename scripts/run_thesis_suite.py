#!/usr/bin/env python3
"""
Run thesis-aligned WireCR-SAM experiment suites and save manifest files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = REPO_ROOT / "main_sam.py"


def add_bool_arg(parser, name, default, help_text):
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return

    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{dest.replace('_', '-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def build_passthrough_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str, default="wire_hole")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--sam-model-type", type=str, default="vit_h")
    parser.add_argument("--head-type", type=str, default="prompt", choices=["prompt", "fpn"])
    parser.add_argument("--train-augment", type=str, default="industrial", choices=["none", "industrial"])
    parser.add_argument("--augment-strength", type=str, default="medium", choices=["light", "medium", "strong"])
    parser.add_argument("--adapter-size", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--adapter-kind", type=str, default="wirecr", choices=["wirecr", "vanilla"])
    parser.add_argument("--compression-ratio", type=int, default=8, choices=[4, 8, 16, 32, 64])
    parser.add_argument("--subset-ratio", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.1)
    parser.add_argument("--cldice-weight", type=float, default=0.1)
    parser.add_argument("--hole-class-weight", type=float, default=2.0)
    parser.add_argument("--fpn-adapter-levels", type=str, default="c4,c5")
    parser.add_argument("--fpn-adapter-size-map", type=str, default=None)
    parser.add_argument("--fpn-compression-map", type=str, default=None)
    parser.add_argument("--fpn-simple-map", type=str, default=None)
    add_bool_arg(parser, "--use-residual", True, "Use residual connection in the WireCR adapter.")
    add_bool_arg(parser, "--adapter-simple", False, "Use a simplified adapter without compression-expansion.")
    add_bool_arg(parser, "--disable-adapter", False, "Disable the WireCR adapter.")
    add_bool_arg(parser, "--class-aware-prompts", True, "Use class-aware prompts.")
    add_bool_arg(parser, "--freeze-encoder", True, "Freeze SAM encoder.")
    add_bool_arg(parser, "--freeze-decoder", False, "Freeze SAM decoder.")
    add_bool_arg(parser, "--freeze-prompt-encoder", True, "Freeze SAM prompt encoder.")
    return parser


def cli_tokens_from_overrides(overrides):
    tokens = []
    for key, value in overrides.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            tokens.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            tokens.extend([flag, str(value)])
    return tokens


def _supports_prompt_controls(base):
    return getattr(base, "head_type", "prompt") == "prompt"


def _fpn_layer_ablation_specs(base):
    if getattr(base, "head_type", "prompt") != "fpn":
        return []

    specs = [
        {
            "row_id": "levels_c5",
            "row_label": "Adapter Levels = c5",
            "description": "Keep adapter only on the deepest FPN level.",
            "overrides": {
                "fpn_adapter_levels": "c5",
                "fpn_adapter_size_map": None,
                "fpn_compression_map": None,
                "fpn_simple_map": None,
            },
        },
        {
            "row_id": "levels_c3_c4_c5",
            "row_label": "Adapter Levels = c3,c4,c5",
            "description": "Add a lightweight low-level adapter on c3 while keeping c4/c5 unchanged.",
            "overrides": {
                "fpn_adapter_levels": "c3,c4,c5",
                "fpn_adapter_size_map": None,
                "fpn_compression_map": None,
                "fpn_simple_map": None,
            },
        },
        {
            "row_id": "levels_c2_c3_c4_c5",
            "row_label": "Adapter Levels = c2,c3,c4,c5",
            "description": "Enable lightweight low-level adapters on both c2 and c3 in addition to c4/c5.",
            "overrides": {
                "fpn_adapter_levels": "c2,c3,c4,c5",
                "fpn_adapter_size_map": None,
                "fpn_compression_map": None,
                "fpn_simple_map": None,
            },
        },
    ]
    base_levels = str(getattr(base, "fpn_adapter_levels", "c4,c5")).replace(" ", "").lower()
    if base_levels != "c4,c5":
        specs.insert(
            1,
            {
                "row_id": "levels_c4_c5",
                "row_label": "Adapter Levels = c4,c5",
                "description": "Baseline FPN injection over the two deepest levels.",
                "overrides": {
                    "fpn_adapter_levels": "c4,c5",
                    "fpn_adapter_size_map": None,
                    "fpn_compression_map": None,
                    "fpn_simple_map": None,
                },
            },
        )
    return specs


def table_4_2_specs(base):
    specs = [
        {
            "row_id": "original_sam_transfer",
            "row_label": "Original SAM Transfer",
            "description": "Disable the WireCR adapter but keep the automatic semantic decoding pipeline.",
            "overrides": {
                "disable_adapter": True,
                "adapter_kind": "wirecr",
                "adapter_simple": False,
            },
        },
        {
            "row_id": "vanilla_adapter",
            "row_label": "Vanilla Adapter",
            "description": "Use a plain bottleneck convolutional adapter baseline instead of WireCR.",
            "overrides": {
                "disable_adapter": False,
                "adapter_kind": "vanilla",
                "adapter_simple": False,
            },
        },
        {
            "row_id": "simple_adapter",
            "row_label": "Simple Adapter",
            "description": "Use the simplified adapter without compression-expansion bottleneck.",
            "overrides": {
                "disable_adapter": False,
                "adapter_kind": "wirecr",
                "adapter_simple": True,
            },
        },
        {
            "row_id": "wirecr_sam",
            "row_label": "WireCR-SAM",
            "description": "Full model with WireCR adapter and class-aware prompts.",
            "overrides": {
                "disable_adapter": False,
                "adapter_kind": "wirecr",
                "adapter_simple": False,
            },
        },
    ]
    if _supports_prompt_controls(base):
        for spec in specs:
            spec["overrides"]["class_aware_prompts"] = True
    return specs


def table_4_3_specs(base):
    specs = [
        {
            "row_id": "reference",
            "row_label": (
                f"Reference ({base.adapter_kind}, {base.adapter_size}, 1/{base.compression_ratio})"
                if getattr(base, "head_type", "prompt") != "fpn"
                else (
                    f"Reference ({base.adapter_kind}, {base.adapter_size}, 1/{base.compression_ratio}, "
                    f"levels={base.fpn_adapter_levels})"
                )
            ),
            "description": "Reference structure used by the main model.",
            "overrides": {},
        }
    ]

    for size in ["small", "medium", "large"]:
        if size == base.adapter_size:
            continue
        specs.append(
            {
                "row_id": f"size_{size}",
                "row_label": f"Adapter Size = {size}",
                "description": "Structure ablation over adapter width.",
                "overrides": {"adapter_kind": "wirecr", "adapter_size": size},
            }
        )

    for ratio in [4, 8, 16, 32, 64]:
        if ratio == base.compression_ratio:
            continue
        specs.append(
            {
                "row_id": f"cr_{ratio}",
                "row_label": f"Compression = 1/{ratio}",
                "description": "Structure ablation over bottleneck compression ratio.",
                "overrides": {"adapter_kind": "wirecr", "compression_ratio": ratio},
            }
        )

    specs.extend(
        [
            {
                "row_id": "adapter_vanilla",
                "row_label": "Adapter Kind = Vanilla",
                "description": "Replace the WireCR adapter with a plain bottleneck baseline.",
                "overrides": {
                    "adapter_kind": "vanilla",
                    "adapter_simple": False,
                },
            },
            {
                "row_id": "adapter_simple",
                "row_label": "Simple Adapter = On",
                "description": "Remove compression-expansion bottleneck.",
                "overrides": {"adapter_kind": "wirecr", "adapter_simple": True},
            },
        ]
    )
    if _supports_prompt_controls(base):
        specs.append(
            {
                "row_id": "class_prompts_off",
                "row_label": "Class-aware Prompts = Off",
                "description": "Disable prompt offsets and keep only learnable class embeddings.",
                "overrides": {"class_aware_prompts": False},
            }
        )
    specs.extend(_fpn_layer_ablation_specs(base))
    return specs


def table_4_4_specs(base):
    return [
        {
            "row_id": "bce_dice",
            "row_label": "BCE + Dice",
            "description": "Basic segmentation losses only.",
            "overrides": {
                "boundary_loss_weight": 0.0,
                "cldice_weight": 0.0,
                "hole_class_weight": 1.0,
            },
        },
        {
            "row_id": "plus_boundary",
            "row_label": "+ Boundary",
            "description": "Add boundary-aware loss on top of BCE + Dice.",
            "overrides": {
                "boundary_loss_weight": base.boundary_loss_weight,
                "cldice_weight": 0.0,
                "hole_class_weight": 1.0,
            },
        },
        {
            "row_id": "plus_cldice",
            "row_label": "+ clDice",
            "description": "Add clDice on top of boundary-aware loss.",
            "overrides": {
                "boundary_loss_weight": base.boundary_loss_weight,
                "cldice_weight": base.cldice_weight,
                "hole_class_weight": 1.0,
            },
        },
        {
            "row_id": "plus_hole_weight",
            "row_label": "+ Hole Weight",
            "description": "Full loss with hole-class weighting.",
            "overrides": {
                "boundary_loss_weight": base.boundary_loss_weight,
                "cldice_weight": base.cldice_weight,
                "hole_class_weight": base.hole_class_weight,
            },
        },
    ]


def table_4_5_specs(base):
    return [
        {
            "row_id": f"subset_{int(ratio * 100):03d}",
            "row_label": f"subset_ratio = {ratio}",
            "description": "Few-shot subset experiment.",
            "overrides": {"subset_ratio": ratio},
        }
        for ratio in (0.1, 0.25, 0.5, 1.0)
    ]


TABLE_BUILDERS = {
    "4-2": ("表 4-2 主对比实验", table_4_2_specs),
    "4-3": ("表 4-3 结构消融实验", table_4_3_specs),
    "4-4": ("表 4-4 损失消融实验", table_4_4_specs),
    "4-5": ("表 4-5 少样本实验", table_4_5_specs),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run thesis-aligned WireCR-SAM experiment suites.")
    parser.add_argument("--table", type=str, default="all", choices=["4-2", "4-3", "4-4", "4-5", "all"])
    parser.add_argument("--output-root", type=str, default="./thesis_runs", help="Directory used to store suite outputs.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch main_sam.py.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments whose structured summaries already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--manifest-name", type=str, default=None, help="Optional custom manifest filename.")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def load_base_config(passthrough):
    parser = build_passthrough_parser()
    base, _ = parser.parse_known_args(passthrough)
    return base


def run_suite(table_id, table_title, spec_builder, suite_args, passthrough, base_config):
    suite_root = Path(suite_args.output_root).resolve() / f"table{table_id.replace('-', '_')}"
    suite_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "table_id": table_id,
        "table_title": table_title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "suite_root": str(suite_root),
        "experiments": [],
    }

    for spec in spec_builder(base_config):
        run_name = f"table{table_id.replace('-', '_')}_{spec['row_id']}"
        run_dir = suite_root / run_name
        summary_path = run_dir / "experiment_summary.json"
        command = [
            suite_args.python,
            str(MAIN_SCRIPT),
            "--mode",
            "sam",
            "--save-dir",
            str(suite_root),
            "--run-name",
            run_name,
            "--results-json",
            str(summary_path),
            *passthrough,
            *cli_tokens_from_overrides(spec["overrides"]),
        ]

        status = "pending"
        if suite_args.skip_existing and summary_path.is_file():
            status = "skipped_existing"
        elif suite_args.dry_run:
            status = "dry_run"
        else:
            subprocess.run(command, check=True, cwd=REPO_ROOT)
            status = "completed"

        manifest["experiments"].append(
            {
                "row_id": spec["row_id"],
                "row_label": spec["row_label"],
                "description": spec["description"],
                "run_name": run_name,
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "overrides": deepcopy(spec["overrides"]),
                "command": command,
                "status": status,
            }
        )

    manifest_name = suite_args.manifest_name or f"manifest_table{table_id.replace('-', '_')}.json"
    manifest_path = suite_root / manifest_name
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"[suite] {table_title} manifest saved to: {manifest_path}")


def main():
    suite_args, passthrough = parse_args()
    base_config = load_base_config(passthrough)

    tables = TABLE_BUILDERS.items() if suite_args.table == "all" else [(suite_args.table, TABLE_BUILDERS[suite_args.table])]
    for table_id, (table_title, spec_builder) in tables:
        run_suite(table_id, table_title, spec_builder, suite_args, passthrough, base_config)


if __name__ == "__main__":
    main()
