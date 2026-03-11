import os
import random
import thop
import torch

from models import crnet
from utils import logger, line_seg

__all__ = ["init_device", "init_model", "init_sam_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_sam_model(args):
    """
    Initialize SAM model with CRNet adapter.

    Args:
        args: Arguments containing SAM and adapter configuration

    Returns:
        SAMWithCRNetAdapter instance
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'sam'))

    try:
        from segment_anything import sam_model_registry
    except ImportError:
        logger.error("SAM not found. Please run: git submodule update --init --recursive")
        logger.error("Or install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
        raise

    # Check if checkpoint is provided
    if args.sam_checkpoint is None:
        logger.warning("No SAM checkpoint provided. Using random initialization.")
        logger.warning("Please download SAM checkpoints from:")
        logger.warning("  ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        logger.warning("  ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
        logger.warning("  ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        raise ValueError("--sam-checkpoint is required for SAM mode")

    if not os.path.isfile(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    # Load SAM model
    logger.info(f'=> Loading SAM model: {args.sam_model_type}')
    logger.info(f'=> Checkpoint: {args.sam_checkpoint}')

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)

    # Create adapter config
    adapter_config = {
        'adapter_size': args.adapter_size,
        'compression_ratio': args.compression_ratio,
        'use_residual': args.use_residual,
        'simple': args.adapter_simple,
    }

    # Wrap SAM with CRNet adapter
    from models.sam_wrapper import SAMWithCRNetAdapter
    model = SAMWithCRNetAdapter(
        sam_model=sam,
        adapter_config=adapter_config,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder,
    )

    # Load pretrained if specified
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
        logger.info("=> SAM+Adapter pretrained model loaded from {}".format(args.pretrained))

    # Model info
    total_params = model.get_num_total_params()
    adapter_params = model.get_num_adapter_params()
    frozen_params = model.get_num_frozen_params()
    trainable_params = total_params - frozen_params

    total_params_fmt = f"{total_params:,}"
    adapter_params_fmt = f"{adapter_params:,}"
    frozen_params_fmt = f"{frozen_params:,}"
    trainable_params_fmt = f"{trainable_params:,}"

    logger.info(f'=> Model Name: SAM + CRNet Adapter')
    logger.info(f'=> SAM Type: {args.sam_model_type}')
    logger.info(f'=> Adapter Config: size={args.adapter_size}, compression=1/{args.compression_ratio}')
    logger.info(f'=> Total Params: {total_params_fmt}')
    logger.info(f'=> Adapter Params: {adapter_params_fmt} ({100*adapter_params/total_params:.2f}%)')
    logger.info(f'=> Frozen Params: {frozen_params_fmt} ({100*frozen_params/total_params:.2f}%)')
    logger.info(f'=> Trainable Params: {trainable_params_fmt}')
    logger.info(f'{line_sep}\n{model}\n{line_sep}\n')

    return model


def init_model(args):
    """
    Initialize model based on mode.

    Args:
        args: Arguments containing mode and model configuration

    Returns:
        Model instance
    """
    if args.mode == 'sam':
        return init_sam_model(args)
    else:
        # Original CSI mode
        # Model loading
        model = crnet(reduction=args.cr)

        if args.pretrained is not None:
            assert os.path.isfile(args.pretrained)
            state_dict = torch.load(args.pretrained,
                                    map_location=torch.device('cpu'))['state_dict']
            model.load_state_dict(state_dict)
            logger.info("pretrained model loaded from {}".format(args.pretrained))

        # Model flops and params counting
        image = torch.randn([1, 2, 32, 32])
        flops, params = thop.profile(model, inputs=(image,), verbose=False)
        flops, params = thop.clever_format([flops, params], "%.3f")

        # Model info logging
        logger.info(f'=> Model Name: CRNet [pretrained: {args.pretrained}]')
        logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
        logger.info(f'=> Model Flops: {flops}')
        logger.info(f'=> Model Params Num: {params}\n')
        logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

        return model
