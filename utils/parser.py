import argparse


def add_bool_arg(parser, name, default, help_text):
    """Add a boolean flag with optional negation."""
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return

    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{dest.replace('_', '-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


parser = argparse.ArgumentParser(description="WireCR-SAM Training")


# ========================== Indispensable arguments ==========================

parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset root.")
parser.add_argument("-b", "--batch-size", type=int, required=True, metavar="N", help="Mini-batch size.")
parser.add_argument("-j", "--workers", type=int, metavar="N", required=True, help="Number of data loading workers.")


# ============================= Mode selection =============================

parser.add_argument(
    "--mode",
    type=str,
    default="sam",
    choices=["sam"],
    help="Retained for backward compatibility. Only SAM mode is supported.",
)


# ============================= SAM-specific arguments =============================

parser.add_argument(
    "--sam-model-type",
    type=str,
    default="vit_h",
    choices=["vit_h", "vit_l", "vit_b"],
    help="SAM model variant.",
)
parser.add_argument(
    "--head-type",
    type=str,
    default="prompt",
    choices=["prompt", "fpn"],
    help="Segmentation head type. prompt keeps the SAM decoder baseline; fpn enables the new parallel FPN head.",
)
parser.add_argument("--sam-checkpoint", type=str, default=None, help="Path to SAM pretrained checkpoint.")
parser.add_argument(
    "--dataset",
    type=str,
    default="wire_hole",
    choices=["wire_hole", "coco"],
    help="Dataset for WireCR-SAM training/evaluation. wire_hole is primary; coco is auxiliary.",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=3,
    help="Number of semantic classes including background. Use 3 for wire_hole and 2 for coco.",
)
parser.add_argument("--image-size", type=int, default=1024, help="Input image size used for training/evaluation.")
parser.add_argument(
    "--subset-ratio",
    type=float,
    default=1.0,
    help="Fraction of training data to use, e.g. 0.1, 0.25, 0.5, or 1.0.",
)
parser.add_argument("--subset-seed", type=int, default=42, help="Seed used when sampling subset_ratio.")
parser.add_argument(
    "--train-augment",
    type=str,
    default="industrial",
    choices=["none", "industrial"],
    help="Training-time augmentation preset. Only wire_hole uses industrial augmentations.",
)
parser.add_argument(
    "--augment-strength",
    type=str,
    default="medium",
    choices=["light", "medium", "strong"],
    help="Strength preset used by the selected training augmentation pipeline.",
)

# Adapter configuration
parser.add_argument(
    "--adapter-size",
    type=str,
    default="medium",
    choices=["small", "medium", "large"],
    help="WireCR adapter size variant.",
)
parser.add_argument(
    "--adapter-kind",
    type=str,
    default="wirecr",
    choices=["wirecr", "vanilla"],
    help="Adapter family used for feature injection. wirecr is the proposed module; vanilla is a plain bottleneck baseline.",
)
parser.add_argument(
    "--compression-ratio",
    type=int,
    default=8,
    choices=[4, 8, 16, 32, 64],
    help="Compression ratio reciprocal for the adapter bottleneck.",
)
add_bool_arg(parser, "--use-residual", True, "Use residual connection in the WireCR adapter.")
add_bool_arg(parser, "--adapter-simple", False, "Use a simplified adapter without compression-expansion.")
add_bool_arg(parser, "--disable-adapter", False, "Disable the WireCR adapter and use raw SAM image embeddings.")
add_bool_arg(parser, "--class-aware-prompts", True, "Use learnable class-aware prompts for automatic decoding.")

# SAM training options
add_bool_arg(parser, "--freeze-encoder", True, "Freeze SAM image encoder weights.")
add_bool_arg(parser, "--freeze-decoder", False, "Freeze SAM mask decoder weights.")
add_bool_arg(parser, "--freeze-prompt-encoder", True, "Freeze SAM prompt encoder weights.")

parser.add_argument("--boundary-loss-weight", type=float, default=0.1, help="Weight for boundary-aware loss.")
parser.add_argument("--cldice-weight", type=float, default=0.1, help="Weight for clDice loss on wire class.")
parser.add_argument(
    "--hole-class-weight",
    type=float,
    default=2.0,
    help="Positive class weight for interface-hole masks.",
)
parser.add_argument("--dice-weight", type=float, default=1.0, help="Weight for Dice loss.")
parser.add_argument("--bce-weight", type=float, default=1.0, help="Weight for BCE loss.")
parser.add_argument(
    "--main-class-weights",
    type=float,
    nargs=3,
    default=[1.0, 1.5, 4.0],
    metavar=("BG", "WIRE", "HOLE"),
    help="Class weights used by the FPN main segmentation head.",
)
parser.add_argument(
    "--hole-aux-weight",
    type=float,
    default=0.3,
    help="Weight applied to the hole auxiliary head loss in the FPN branch.",
)
parser.add_argument(
    "--fpn-adapter-levels",
    type=str,
    default="c4,c5",
    help=(
        "Comma-separated FPN feature levels that use WireCR adapters, e.g. c4,c5 or c2,c3,c4,c5. "
        "If c2/c3 are enabled without explicit per-level overrides, they use the recommended "
        "lightweight preset (small + cr16 + simple)."
    ),
)
parser.add_argument(
    "--fpn-adapter-size-map",
    type=str,
    default=None,
    help="Optional per-level adapter-size overrides, e.g. c2=small,c3=small,c4=medium,c5=medium.",
)
parser.add_argument(
    "--fpn-compression-map",
    type=str,
    default=None,
    help="Optional per-level compression-ratio overrides, e.g. c2=16,c3=16,c4=8,c5=8.",
)
parser.add_argument(
    "--fpn-simple-map",
    type=str,
    default=None,
    help="Optional per-level simple/full overrides, e.g. c2=1,c3=1,c4=0,c5=0.",
)


# ============================= Optional arguments =============================

parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="Evaluate model on validation/test set.")
parser.add_argument("--pretrained", type=str, default=None, help="Path to a pre-trained model checkpoint.")
parser.add_argument("--resume", type=str, metavar="PATH", default=None, help="Path to latest checkpoint.")
parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Root directory used to save run outputs.")
parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run name used under save-dir.")
parser.add_argument(
    "--results-json",
    type=str,
    default=None,
    help=(
        "Optional structured JSON output. If a bare name is provided, it is saved next to "
        "--pretrained/--resume as <name>.json; otherwise the explicit path is used."
    ),
)
parser.add_argument("--seed", default=None, type=int, help="Seed for initialization.")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--cpu", action="store_true", help="Disable GPU training.")
parser.add_argument("--cpu-affinity", default=None, type=str, help='CPU affinity, like "0xffff".')
parser.add_argument("--epochs", type=int, metavar="N", help="Number of total epochs to run.")
parser.add_argument("--scheduler", type=str, default="const", choices=["const", "cosine"], help="Learning rate scheduler.")
parser.add_argument(
    "--val-freq",
    type=int,
    default=1,
    help="Run validation every N epochs during training.",
)
parser.add_argument(
    "--test-freq",
    type=int,
    default=1,
    help="Run test evaluation every N epochs during training.",
)

args = parser.parse_args()
