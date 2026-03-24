#!/usr/bin/env python3
"""
Export thesis-ready CSV tables from structured WireCR-SAM experiment summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_summary_path(path_like):
    path = Path(path_like).resolve()
    if path.is_dir():
        candidate = path / "experiment_summary.json"
        if candidate.is_file():
            return candidate
    if path.is_file():
        return path
    raise FileNotFoundError(f"Could not resolve experiment summary: {path_like}")


def load_summary(path_like):
    return load_json(resolve_summary_path(path_like))


def load_manifest(path_like):
    manifest_path = Path(path_like).resolve()
    manifest = load_json(manifest_path)
    experiments = []
    for item in manifest.get("experiments", []):
        summary_path = Path(item["summary_path"]).resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing experiment summary for row '{item['row_id']}': {summary_path}")
        experiments.append((item, load_json(summary_path)))
    return manifest, experiments


def write_csv(output_path, fieldnames, rows):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] CSV saved to: {output_path}")


def get_hole_metric(results, suffix):
    return results.get(f"hole_{suffix}", results.get(f"interface_hole_{suffix}"))


def export_table_4_1(args):
    summary = load_summary(args.reference)
    config = summary["config"]
    model = summary["model"]
    row = {
        "sam_model_type": config["sam_model_type"],
        "adapter_variant": model["adapter_variant"],
        "adapter_kind": config.get("adapter_kind", "wirecr"),
        "adapter_size": config["adapter_size"],
        "compression_ratio": config["compression_ratio"],
        "freeze_encoder": int(bool(config["freeze_encoder"])),
        "freeze_decoder": int(bool(config["freeze_decoder"])),
        "freeze_prompt_encoder": int(bool(config["freeze_prompt_encoder"])),
        "class_aware_prompts": int(bool(config["class_aware_prompts"])),
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "epochs": config["epochs"],
        "scheduler": config["scheduler"],
        "subset_ratio": config["subset_ratio"],
        "bce_weight": config["bce_weight"],
        "dice_weight": config["dice_weight"],
        "boundary_loss_weight": config["boundary_loss_weight"],
        "cldice_weight": config["cldice_weight"],
        "hole_class_weight": config["hole_class_weight"],
        "total_params": model["total_params"],
        "trainable_params": model["trainable_params"],
    }
    fieldnames = list(row.keys())
    write_csv(args.output, fieldnames, [row])


def export_table_4_2(args):
    _, experiments = load_manifest(args.manifest)
    rows = []
    for item, summary in experiments:
        results = summary["results"]
        rows.append(
            {
                "Model": item["row_label"],
                "IoU": results.get("iou"),
                "Dice": results.get("dice"),
                "Precision": results.get("precision"),
                "Recall": results.get("recall"),
                "F1": results.get("f1"),
                "Boundary F1": results.get("boundary_f1"),
                "clDice": results.get("cldice"),
                "Wire IoU": results.get("wire_iou"),
                "Hole IoU": get_hole_metric(results, "iou"),
                "Hole Recall": get_hole_metric(results, "recall"),
            }
        )
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.output, fieldnames, rows)


def export_table_4_3(args):
    _, experiments = load_manifest(args.manifest)
    rows = []
    for item, summary in experiments:
        config = summary["config"]
        results = summary["results"]
        rows.append(
            {
                "Experiment": item["row_label"],
                "adapter kind": config.get("adapter_kind", "wirecr"),
                "adapter size": config["adapter_size"],
                "compression ratio": config["compression_ratio"],
                "adapter simple": int(bool(config["adapter_simple"])),
                "class-aware prompts": int(bool(config["class_aware_prompts"])),
                "fpn adapter levels": config.get("fpn_adapter_levels", "c4,c5"),
                "IoU": results.get("iou"),
                "Dice": results.get("dice"),
                "Boundary F1": results.get("boundary_f1"),
                "clDice": results.get("cldice"),
                "Hole Recall": get_hole_metric(results, "recall"),
            }
        )
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.output, fieldnames, rows)


def export_table_4_4(args):
    _, experiments = load_manifest(args.manifest)
    rows = []
    for item, summary in experiments:
        config = summary["config"]
        results = summary["results"]
        rows.append(
            {
                "Experiment": item["row_label"],
                "BCE loss": int(float(config["bce_weight"]) > 0),
                "Dice loss": int(float(config["dice_weight"]) > 0),
                "Boundary loss": int(float(config["boundary_loss_weight"]) > 0),
                "clDice loss": int(float(config["cldice_weight"]) > 0),
                "hole weight": config["hole_class_weight"],
                "IoU": results.get("iou"),
                "Dice metric": results.get("dice"),
                "Boundary F1": results.get("boundary_f1"),
                "clDice metric": results.get("cldice"),
                "Hole IoU": get_hole_metric(results, "iou"),
                "Hole Recall": get_hole_metric(results, "recall"),
            }
        )
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.output, fieldnames, rows)


def export_table_4_5(args):
    _, experiments = load_manifest(args.manifest)
    rows = []
    for item, summary in experiments:
        config = summary["config"]
        dataset = summary["dataset"]
        results = summary["results"]
        rows.append(
            {
                "subset ratio": config["subset_ratio"],
                "train samples": dataset["train_samples"],
                "IoU": results.get("iou"),
                "Dice": results.get("dice"),
                "Wire IoU": results.get("wire_iou"),
                "Hole IoU": get_hole_metric(results, "iou"),
                "Hole Recall": get_hole_metric(results, "recall"),
                "Boundary F1": results.get("boundary_f1"),
                "clDice": results.get("cldice"),
            }
        )
    rows.sort(key=lambda row: float(row["subset ratio"]))
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.output, fieldnames, rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Export thesis-ready CSV tables.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p41 = subparsers.add_parser("table4_1", help="Export 表 4-1 training configuration CSV.")
    p41.add_argument("--reference", required=True, help="Experiment summary JSON or run directory.")
    p41.add_argument("--output", default="./thesis_tables/table4_1.csv")
    p41.set_defaults(func=export_table_4_1)

    p42 = subparsers.add_parser("table4_2", help="Export 表 4-2 comparison CSV.")
    p42.add_argument("--manifest", required=True, help="Manifest JSON produced by run_thesis_suite.py.")
    p42.add_argument("--output", default="./thesis_tables/table4_2.csv")
    p42.set_defaults(func=export_table_4_2)

    p43 = subparsers.add_parser("table4_3", help="Export 表 4-3 structure ablation CSV.")
    p43.add_argument("--manifest", required=True, help="Manifest JSON produced by run_thesis_suite.py.")
    p43.add_argument("--output", default="./thesis_tables/table4_3.csv")
    p43.set_defaults(func=export_table_4_3)

    p44 = subparsers.add_parser("table4_4", help="Export 表 4-4 loss ablation CSV.")
    p44.add_argument("--manifest", required=True, help="Manifest JSON produced by run_thesis_suite.py.")
    p44.add_argument("--output", default="./thesis_tables/table4_4.csv")
    p44.set_defaults(func=export_table_4_4)

    p45 = subparsers.add_parser("table4_5", help="Export 表 4-5 few-shot CSV.")
    p45.add_argument("--manifest", required=True, help="Manifest JSON produced by run_thesis_suite.py.")
    p45.add_argument("--output", default="./thesis_tables/table4_5.csv")
    p45.set_defaults(func=export_table_4_5)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
