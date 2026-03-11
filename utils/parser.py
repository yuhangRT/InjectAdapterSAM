import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--data-dir', type=str, required=True,
                    help='the path of dataset.')
parser.add_argument('-b', '--batch-size', type=int, required=True, metavar='N',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')


# ============================= Mode selection =============================

parser.add_argument('--mode', type=str, default='csi',
                    choices=['csi', 'sam'],
                    help='Operation mode: csi for CSI feedback, sam for SAM adapter')


# ============================= CSI-specific arguments =============================

parser.add_argument('--scenario', type=str, default=None, choices=["in", "out"],
                    help="the channel scenario (for CSI mode only)")


# ============================= SAM-specific arguments =============================

parser.add_argument('--sam-model-type', type=str, default='vit_h',
                    choices=['vit_h', 'vit_l', 'vit_b'],
                    help='SAM model variant (for SAM mode only)')
parser.add_argument('--sam-checkpoint', type=str, default=None,
                    help='Path to SAM pretrained checkpoint (for SAM mode only)')
parser.add_argument('--dataset', type=str, default='coco',
                    choices=['coco', 'custom'],
                    help='Dataset for SAM training/evaluation')

# Adapter configuration
parser.add_argument('--adapter-size', type=str, default='medium',
                    choices=['small', 'medium', 'large'],
                    help='CRNet adapter size variant')
parser.add_argument('--compression-ratio', type=int, default=4,
                    choices=[4, 8, 16, 32, 64],
                    help='Compression ratio (reciprocal)')
parser.add_argument('--use-residual', action='store_true', default=True,
                    help='Use residual connection in adapter')
parser.add_argument('--adapter-simple', action='store_true', default=False,
                    help='Use simplified adapter without FC compression')

# SAM training options
parser.add_argument('--freeze-encoder', action='store_true', default=True,
                    help='Freeze SAM encoder weights')
parser.add_argument('--freeze-decoder', action='store_true', default=False,
                    help='Freeze SAM decoder weights')
parser.add_argument('--freeze-prompt-encoder', action='store_true', default=True,
                    help='Freeze SAM prompt encoder weights')
parser.add_argument('--prompt-strategy', type=str, default='random',
                    choices=['random', 'center', 'grid', 'box'],
                    help='Prompt generation strategy for SAM training')
parser.add_argument('--num-prompts', type=int, default=1,
                    help='Number of prompts per image during SAM training')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio (for CSI mode only)')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')

args = parser.parse_args()
