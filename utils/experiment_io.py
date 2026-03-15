import json
import os
import re
from copy import deepcopy
from datetime import datetime


__all__ = [
    "build_run_name",
    "resolve_run_dir",
    "write_json",
    "build_experiment_summary",
]


def _format_float_tag(value):
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def _sanitize_tag(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "run"


def build_run_name(args):
    if getattr(args, "run_name", None):
        return _sanitize_tag(args.run_name)

    adapter_variant = "noadapter" if args.disable_adapter else ("simple" if args.adapter_simple else "full")
    parts = [
        "wirecrsam",
        _sanitize_tag(args.dataset),
        _sanitize_tag(args.sam_model_type),
        adapter_variant,
    ]

    if not args.disable_adapter:
        parts.extend(
            [
                _sanitize_tag(args.adapter_size),
                f"cr{args.compression_ratio}",
            ]
        )

    parts.extend(
        [
            f"cap{int(bool(args.class_aware_prompts))}",
            f"sr{int(round(float(args.subset_ratio) * 100)):03d}",
        ]
    )

    if not args.freeze_encoder:
        parts.append("fe0")
    if args.freeze_decoder:
        parts.append("fd1")
    if not args.freeze_prompt_encoder:
        parts.append("fp0")
    if not args.use_residual:
        parts.append("res0")
    if float(args.boundary_loss_weight) != 0.1:
        parts.append(f"bw{_format_float_tag(args.boundary_loss_weight)}")
    if float(args.cldice_weight) != 0.1:
        parts.append(f"cw{_format_float_tag(args.cldice_weight)}")
    if float(args.hole_class_weight) != 2.0:
        parts.append(f"hw{_format_float_tag(args.hole_class_weight)}")

    return "_".join(parts)


def resolve_run_dir(args):
    explicit_dir = bool(getattr(args, "run_name", None)) or getattr(args, "save_dir", "./checkpoints") != "./checkpoints"
    if getattr(args, "evaluate", False) and getattr(args, "pretrained", None) and not explicit_dir:
        pretrained_dir = os.path.dirname(os.path.abspath(args.pretrained))
        if pretrained_dir:
            return pretrained_dir

    run_name = build_run_name(args)
    return os.path.join(os.path.abspath(args.save_dir), run_name)


def _to_serializable(value):
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def build_experiment_summary(
    *,
    args,
    run_dir,
    class_names,
    train_samples,
    val_samples,
    test_samples,
    model,
    results,
    best_iou=None,
    checkpoint_paths=None,
    stage,
):
    total_params = model.get_num_total_params()
    adapter_params = model.get_num_adapter_params()
    frozen_params = model.get_num_frozen_params()

    config = {
        "mode": args.mode,
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "epochs": args.epochs,
        "scheduler": args.scheduler,
        "sam_model_type": args.sam_model_type,
        "sam_checkpoint": args.sam_checkpoint,
        "pretrained": args.pretrained,
        "resume": args.resume,
        "subset_ratio": args.subset_ratio,
        "subset_seed": args.subset_seed,
        "adapter_size": args.adapter_size,
        "compression_ratio": args.compression_ratio,
        "use_residual": args.use_residual,
        "adapter_simple": args.adapter_simple,
        "disable_adapter": args.disable_adapter,
        "class_aware_prompts": args.class_aware_prompts,
        "freeze_encoder": args.freeze_encoder,
        "freeze_decoder": args.freeze_decoder,
        "freeze_prompt_encoder": args.freeze_prompt_encoder,
        "bce_weight": args.bce_weight,
        "dice_weight": args.dice_weight,
        "boundary_loss_weight": args.boundary_loss_weight,
        "cldice_weight": args.cldice_weight,
        "hole_class_weight": args.hole_class_weight,
        "save_dir": args.save_dir,
        "run_name": build_run_name(args),
    }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "run_dir": os.path.abspath(run_dir),
        "config": deepcopy(config),
        "dataset": {
            "class_names": list(class_names),
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "test_samples": int(test_samples),
        },
        "model": {
            "adapter_variant": "none" if args.disable_adapter else ("simple" if args.adapter_simple else "full"),
            "total_params": int(total_params),
            "adapter_params": int(adapter_params),
            "frozen_params": int(frozen_params),
            "trainable_params": int(total_params - frozen_params),
        },
        "results": _to_serializable(results),
        "best_iou": None if best_iou is None else float(best_iou),
        "checkpoints": _to_serializable(checkpoint_paths or {}),
    }
    return summary
