import os
import random

import torch
from utils import logger, line_seg

__all__ = ["init_device", "init_sam_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    if affinity is not None:
        os.system(f"taskset -p {affinity} {os.getpid()}")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if not cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu is not None else 0))
    else:
        pin_memory = False
        device = torch.device("cpu")
        logger.info("Running on CPU")

    return device, pin_memory


def init_sam_model(args):
    """Initialize WireCR-SAM."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "sam"))

    try:
        from segment_anything import sam_model_registry
    except ImportError:
        logger.error("SAM not found. Please run: git submodule update --init --recursive")
        logger.error("Or install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
        raise

    if args.sam_checkpoint is None:
        raise ValueError("--sam-checkpoint is required for SAM mode")
    if not os.path.isfile(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    logger.info(f"=> Loading SAM model: {args.sam_model_type}")
    logger.info(f"=> Checkpoint: {args.sam_checkpoint}")

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)

    adapter_config = {
        "adapter_size": args.adapter_size,
        "compression_ratio": args.compression_ratio,
        "use_residual": args.use_residual,
        "simple": args.adapter_simple,
    }

    from models.sam_wrapper import SAMWithCRNetAdapter

    class_names = (
        ["background", "foreground"]
        if args.dataset == "coco"
        else ["background", "wire", "interface-hole"]
    )

    model = SAMWithCRNetAdapter(
        sam_model=sam,
        adapter_config=adapter_config,
        num_classes=args.num_classes,
        class_names=class_names[:args.num_classes],
        disable_adapter=args.disable_adapter,
        class_aware_prompts=args.class_aware_prompts,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder,
    )

    if args.pretrained is not None and os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict["state_dict"] if "state_dict" in state_dict else state_dict)
        logger.info(f"=> WireCR-SAM pretrained model loaded from {args.pretrained}")

    total_params = model.get_num_total_params()
    adapter_params = model.get_num_adapter_params()
    frozen_params = model.get_num_frozen_params()
    trainable_params = total_params - frozen_params

    adapter_variant = "none" if args.disable_adapter else ("simple" if args.adapter_simple else "full")

    logger.info("=> Model Name: WireCR-SAM")
    logger.info(f"=> SAM Type: {args.sam_model_type}")
    logger.info(
        f"=> Adapter Config: variant={adapter_variant}, size={args.adapter_size}, "
        f"compression=1/{args.compression_ratio}, class_prompts={args.class_aware_prompts}"
    )
    logger.info(f"=> Semantic Classes: {args.num_classes}")
    logger.info(f"=> Class Names: {', '.join(model.class_names)}")
    logger.info(f"=> Total Params: {total_params:,}")
    logger.info(f"=> Adapter Params: {adapter_params:,} ({100 * adapter_params / total_params:.2f}%)")
    logger.info(f"=> Frozen Params: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    logger.info(f"=> Trainable Params: {trainable_params:,}")
    logger.info(f"{line_seg}\n{model}\n{line_seg}\n")

    return model
