import json
import os
import re
from copy import deepcopy
from datetime import datetime


__all__ = [
    "build_run_name",
    "resolve_run_dir",
    "resolve_results_json_path",
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

    if args.disable_adapter:
        adapter_variant = "noadapter"
    elif getattr(args, "adapter_kind", "wirecr") == "vanilla":
        adapter_variant = "vanilla"
    else:
        adapter_variant = "simple" if args.adapter_simple else "full"
    parts = [
        "wirecrsam",
        _sanitize_tag(args.dataset),
        _sanitize_tag(args.sam_model_type),
        _sanitize_tag(args.head_type) if getattr(args, "head_type", "prompt") != "prompt" else None,
        adapter_variant,
    ]
    parts = [part for part in parts if part is not None]

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

    if getattr(args, "train_augment", "industrial") != "industrial":
        parts.append(f"aug{_sanitize_tag(args.train_augment)}")
    else:
        parts.append(f"augind-{_sanitize_tag(getattr(args, 'augment_strength', 'medium'))}")

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
    if getattr(args, "head_type", "prompt") == "fpn":
        normalized_levels = _sanitize_tag(getattr(args, "fpn_adapter_levels", "c4,c5")).replace("-", "")
        if normalized_levels and normalized_levels != "c4c5":
            parts.append(f"lvl{normalized_levels}")
        if any(
            getattr(args, name, None)
            for name in ("fpn_adapter_size_map", "fpn_compression_map", "fpn_simple_map")
        ):
            parts.append("lvlmap")

    return "_".join(parts)


def resolve_run_dir(args):
    explicit_dir = bool(getattr(args, "run_name", None)) or getattr(args, "save_dir", "./checkpoints") != "./checkpoints"
    if getattr(args, "evaluate", False) and getattr(args, "pretrained", None) and not explicit_dir:
        pretrained_dir = os.path.dirname(os.path.abspath(args.pretrained))
        if pretrained_dir:
            return pretrained_dir

    run_name = build_run_name(args)
    return os.path.join(os.path.abspath(args.save_dir), run_name)


def resolve_results_json_path(args, run_dir):
    default_file_name = "evaluate.json" if getattr(args, "evaluate", False) else "experiment_summary.json"
    raw_value = getattr(args, "results_json", None)
    if not raw_value:
        return os.path.join(run_dir, default_file_name)

    raw_value = str(raw_value).strip()
    if not raw_value:
        return os.path.join(run_dir, default_file_name)

    has_explicit_dir = os.path.isabs(raw_value) or bool(os.path.dirname(raw_value))
    if has_explicit_dir:
        return raw_value if raw_value.endswith(".json") else f"{raw_value}.json"

    base_dir = None
    if getattr(args, "pretrained", None):
        base_dir = os.path.dirname(os.path.abspath(args.pretrained))
    elif getattr(args, "resume", None):
        base_dir = os.path.dirname(os.path.abspath(args.resume))

    if not base_dir:
        base_dir = os.path.abspath(run_dir)

    file_name = raw_value if raw_value.endswith(".json") else f"{raw_value}.json"
    return os.path.join(base_dir, file_name)


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
    effective_train_augment = "none" if getattr(args, "evaluate", False) or args.dataset != "wire_hole" else args.train_augment

    config = {
        "mode": args.mode,
        "dataset": args.dataset,
        "head_type": getattr(args, "head_type", "prompt"),
        "num_classes": args.num_classes,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "epochs": args.epochs,
        "scheduler": args.scheduler,
        "val_freq": args.val_freq,
        "test_freq": args.test_freq,
        "sam_model_type": args.sam_model_type,
        "sam_checkpoint": args.sam_checkpoint,
        "pretrained": args.pretrained,
        "resume": args.resume,
        "subset_ratio": args.subset_ratio,
        "subset_seed": args.subset_seed,
        "train_augment": effective_train_augment,
        "augment_strength": args.augment_strength if effective_train_augment != "none" else None,
        "adapter_size": args.adapter_size,
        "adapter_kind": getattr(args, "adapter_kind", "wirecr"),
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
        "main_class_weights": list(getattr(args, "main_class_weights", [1.0, 1.5, 4.0])),
        "hole_aux_weight": getattr(args, "hole_aux_weight", 0.3),
        "fpn_adapter_levels": getattr(args, "fpn_adapter_levels", "c4,c5"),
        "fpn_adapter_size_map": getattr(args, "fpn_adapter_size_map", None),
        "fpn_compression_map": getattr(args, "fpn_compression_map", None),
        "fpn_simple_map": getattr(args, "fpn_simple_map", None),
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
            "head_type": getattr(args, "head_type", "prompt"),
            "adapter_variant": (
                "none"
                if args.disable_adapter
                else ("vanilla" if getattr(args, "adapter_kind", "wirecr") == "vanilla" else ("simple" if args.adapter_simple else "full"))
            ),
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
