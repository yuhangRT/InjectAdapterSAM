#!/usr/bin/env python3
"""
Run single-image inference for WireCR-SAM and save the predicted segmentation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WIRE_HOLE_CLASS_NAMES = ["background", "wire", "interface-hole"]
WIRE_HOLE_PALETTE = [
    0, 0, 0,
    0, 255, 0,
    255, 0, 0,
] + [0, 0, 0] * 253


def add_bool_arg(parser, name, default, help_text):
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return

    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{dest.replace('_', '-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def build_parser():
    parser = argparse.ArgumentParser(description="Single-image inference for WireCR-SAM.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--pretrained", required=True, help="Path to WireCR-SAM checkpoint (best_iou.pth or last.pth).")
    parser.add_argument("--sam-checkpoint", default=None, help="Path to SAM pretrained checkpoint.")
    parser.add_argument("--summary-json", default=None, help="Optional experiment_summary.json used to infer model config.")
    parser.add_argument("--output-dir", default="./inference_outputs", help="Directory used to save inference results.")
    parser.add_argument("--image-size", type=int, default=None, help="Inference resize size. Defaults to training config or 1024.")
    parser.add_argument("--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--adapter-size", choices=["small", "medium", "large"], default=None)
    parser.add_argument("--compression-ratio", type=int, choices=[4, 8, 16, 32, 64], default=None)
    parser.add_argument("--gpu", type=int, default=None, help="GPU id to use.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    add_bool_arg(parser, "--use-residual", True, "Use residual connection in the WireCR adapter.")
    add_bool_arg(parser, "--adapter-simple", False, "Use simplified adapter.")
    add_bool_arg(parser, "--disable-adapter", False, "Disable the adapter.")
    add_bool_arg(parser, "--class-aware-prompts", True, "Use class-aware prompt offsets.")
    add_bool_arg(parser, "--save-overlay", True, "Save a blended overlay image.")
    add_bool_arg(parser, "--save-mask", True, "Save the raw palette mask PNG.")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay blend weight for the predicted mask.",
    )
    return parser


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_summary_path(args):
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(f"Summary JSON not found: {summary_path}")
        return summary_path

    checkpoint_path = Path(args.pretrained).resolve()
    candidate = checkpoint_path.parent / "experiment_summary.json"
    if candidate.is_file():
        return candidate
    return None


def resolve_config(args):
    summary_path = resolve_summary_path(args)
    summary_config = {}
    if summary_path is not None:
        summary = load_json(summary_path)
        summary_config = summary.get("config", {})

    def choose(name, default=None):
        value = getattr(args, name)
        if value is not None:
            return value
        if name in summary_config:
            return summary_config[name]
        return default

    config = {
        "sam_model_type": choose("sam_model_type"),
        "sam_checkpoint": choose("sam_checkpoint"),
        "num_classes": int(choose("num_classes", 3)),
        "image_size": int(choose("image_size", 1024)),
        "adapter_size": choose("adapter_size", "medium"),
        "compression_ratio": int(choose("compression_ratio", 8)),
        "use_residual": choose("use_residual", True),
        "adapter_simple": choose("adapter_simple", False),
        "disable_adapter": choose("disable_adapter", False),
        "class_aware_prompts": choose("class_aware_prompts", True),
    }

    missing = [key for key in ("sam_model_type", "sam_checkpoint") if not config[key]]
    if missing:
        raise ValueError(
            f"Missing required model config values: {missing}. "
            "Pass them explicitly or provide --summary-json / a checkpoint directory with experiment_summary.json."
        )

    return config


def build_device(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config, checkpoint_path, device):
    sam_repo = REPO_ROOT / "third_party" / "sam"
    if str(sam_repo) not in sys.path:
        sys.path.insert(0, str(sam_repo))

    from segment_anything import sam_model_registry
    from models.sam_wrapper import SAMWithCRNetAdapter

    sam = sam_model_registry[config["sam_model_type"]](checkpoint=config["sam_checkpoint"])
    model = SAMWithCRNetAdapter(
        sam_model=sam,
        adapter_config={
            "adapter_size": config["adapter_size"],
            "compression_ratio": config["compression_ratio"],
            "use_residual": config["use_residual"],
            "simple": config["adapter_simple"],
        },
        num_classes=config["num_classes"],
        class_names=WIRE_HOLE_CLASS_NAMES[: config["num_classes"]],
        disable_adapter=config["disable_adapter"],
        class_aware_prompts=config["class_aware_prompts"],
        freeze_encoder=True,
        freeze_decoder=False,
        freeze_prompt_encoder=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size[::-1]
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    return image, tensor, original_size


def infer_mask(model, image_tensor, output_size):
    with torch.no_grad():
        outputs = model(
            [
                {
                    "image": image_tensor,
                    "original_size": tuple(image_tensor.shape[-2:]),
                    "output_size": output_size,
                }
            ],
            multimask_output=False,
        )
    return outputs["semantic_masks"][0].detach().cpu().numpy().astype(np.uint8)


def build_palette_mask(mask_array):
    mask = Image.fromarray(mask_array, mode="P")
    mask.putpalette(WIRE_HOLE_PALETTE)
    return mask


def build_overlay(base_image, mask_array, alpha):
    overlay = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    color_mask = np.zeros_like(overlay)
    color_mask[mask_array == 1] = np.array([0, 255, 0], dtype=np.float32)
    color_mask[mask_array == 2] = np.array([255, 0, 0], dtype=np.float32)

    foreground = mask_array > 0
    blended = overlay.copy()
    blended[foreground] = (1.0 - alpha) * overlay[foreground] + alpha * color_mask[foreground]
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")


def summarize_mask(mask_array):
    values, counts = np.unique(mask_array, return_counts=True)
    summary = []
    for value, count in zip(values.tolist(), counts.tolist()):
        class_name = WIRE_HOLE_CLASS_NAMES[value] if value < len(WIRE_HOLE_CLASS_NAMES) else f"class_{value}"
        summary.append((value, class_name, count))
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    checkpoint_path = Path(args.pretrained).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = resolve_config(args)
    device = build_device(args)
    model = load_model(config, str(checkpoint_path), device)

    base_image, image_tensor, original_size = prepare_image(image_path, config["image_size"])
    pred_mask = infer_mask(model, image_tensor, original_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    mask_path = output_dir / f"{stem}_pred_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    if args.save_mask:
        palette_mask = build_palette_mask(pred_mask)
        palette_mask.save(mask_path)
        print(f"[save] mask: {mask_path}")

    if args.save_overlay:
        overlay = build_overlay(base_image, pred_mask, alpha=args.overlay_alpha)
        overlay.save(overlay_path)
        print(f"[save] overlay: {overlay_path}")

    print(f"[info] device: {device}")
    print(f"[info] image: {image_path}")
    print(f"[info] checkpoint: {checkpoint_path}")
    print(f"[info] sam_model_type: {config['sam_model_type']}")
    print(f"[info] adapter: {'disabled' if config['disable_adapter'] else ('simple' if config['adapter_simple'] else 'full')}")
    print(f"[info] image_size: {config['image_size']}")

    for value, class_name, count in summarize_mask(pred_mask):
        print(f"[mask] label={value} class={class_name} pixels={count}")


if __name__ == "__main__":
    main()
