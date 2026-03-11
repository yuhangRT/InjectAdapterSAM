"""
SAM dataset loaders with prompt generation.

This module provides dataset loaders compatible with SAM's training and
evaluation format, including automatic prompt generation from ground truth masks.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pycocotools.coco as coco
from pycocotools import mask as coco_mask

__all__ = ['SAMDataset', 'SAMPromptGenerator']


class SAMPromptGenerator:
    """
    Generate training prompts (points, boxes) from ground truth masks.

    This class provides static methods to generate various types of prompts
    that SAM uses during training and inference.
    """

    @staticmethod
    def random_point(mask, num_points=1):
        """
        Sample random points from the foreground of a mask.

        Args:
            mask: (H, W) binary mask tensor or numpy array
            num_points: Number of points to sample

        Returns:
            point_coords: (num_points, 2) array of [y, x] coordinates
            point_labels: (num_points,) array of labels (1 for foreground)
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Get foreground coordinates
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            # No foreground, return random background points
            h, w = mask.shape
            point_coords = np.random.rand(num_points, 2) * [h, w]
            point_labels = np.zeros(num_points, dtype=np.int32)
            return point_coords, point_labels

        # Sample random foreground points
        indices = np.random.choice(len(y_coords), size=min(num_points, len(y_coords)), replace=False)

        point_coords = np.stack([y_coords[indices], x_coords[indices]], axis=1).astype(np.float32)
        point_labels = np.ones(num_points, dtype=np.int32)

        # If we need more points than available foreground pixels, add background points
        if num_points > len(y_coords):
            remaining = num_points - len(y_coords)
            bg_y = np.random.randint(0, mask.shape[0], size=remaining)
            bg_x = np.random.randint(0, mask.shape[1], size=remaining)
            bg_coords = np.stack([bg_y, bg_x], axis=1).astype(np.float32)
            bg_labels = np.zeros(remaining, dtype=np.int32)

            point_coords = np.concatenate([point_coords, bg_coords], axis=0)
            point_labels = np.concatenate([point_labels, bg_labels], axis=0)

        return point_coords, point_labels

    @staticmethod
    def center_point(mask):
        """
        Get the center point of the mask.

        Args:
            mask: (H, W) binary mask

        Returns:
            point_coords: (1, 2) array of [y, x] center coordinates
            point_labels: (1,) array with label 1
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Get bounding box
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            # No foreground, return image center
            h, w = mask.shape
            point_coords = np.array([[h // 2, w // 2]], dtype=np.float32)
        else:
            # Compute center of mass
            center_y = y_coords.mean()
            center_x = x_coords.mean()
            point_coords = np.array([[center_y, center_x]], dtype=np.float32)

        point_labels = np.ones(1, dtype=np.int32)

        return point_coords, point_labels

    @staticmethod
    def bounding_box(mask):
        """
        Get the tight bounding box around a mask.

        Args:
            mask: (H, W) binary mask

        Returns:
            box: (4,) array of [x1, y1, x2, y2] in pixel coordinates
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            # No foreground, return full image
            h, w = mask.shape
            return np.array([0, 0, w, h], dtype=np.float32)

        y1, y2 = y_coords.min(), y_coords.max()
        x1, x2 = x_coords.min(), x_coords.max()

        # Add small padding
        padding = 1
        y1 = max(0, y1 - padding)
        x1 = max(0, x1 - padding)
        y2 = min(mask.shape[0], y2 + padding + 1)
        x2 = min(mask.shape[1], x2 + padding + 1)

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def grid_points(mask, grid_spacing=16):
        """
        Sample points on a regular grid.

        Args:
            mask: (H, W) binary mask
            grid_spacing: Spacing between grid points

        Returns:
            point_coords: (N, 2) array of grid points
            point_labels: (N,) array of labels (1 for foreground, 0 for background)
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        h, w = mask.shape
        y_coords = np.arange(grid_spacing // 2, h, grid_spacing)
        x_coords = np.arange(grid_spacing // 2, w, grid_spacing)

        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        point_coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)

        # Label based on mask
        point_labels = mask[point_coords[:, 0].astype(np.int32),
                           point_coords[:, 1].astype(np.int32)].astype(np.int32)

        return point_coords, point_labels


class SAMDataset(Dataset):
    """
    Generic dataset for SAM training/evaluation.

    Supports multiple datasets including COCO and custom datasets.

    Args:
        data_root: Root directory of the dataset
        split: 'train', 'val', or 'test'
        prompt_strategy: 'random', 'center', 'grid', or 'box'
        num_prompts: Number of prompts per image
        transform: Optional transform for images
        image_size: Size to resize images to

    Expected data structure:
        For COCO:
        data_root/
            images/
                train2017/
                val2017/
            annotations/
                instances_train2017.json
                instances_val2017.json
    """

    def __init__(self, data_root, split='train', dataset_name='coco',
                 prompt_strategy='random', num_prompts=1,
                 transform=None, image_size=1024):
        super(SAMDataset, self).__init__()

        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.prompt_strategy = prompt_strategy
        self.num_prompts = num_prompts
        self.transform = transform
        self.image_size = image_size

        # Load dataset
        if dataset_name == 'coco':
            self._load_coco()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _load_coco(self):
        """Load COCO dataset."""
        # Set paths
        if self.split == 'train':
            image_dir = os.path.join(self.data_root, 'images', 'train2017')
            ann_file = os.path.join(self.data_root, 'annotations', 'instances_train2017.json')
        elif self.split == 'val':
            image_dir = os.path.join(self.data_root, 'images', 'val2017')
            ann_file = os.path.join(self.data_root, 'annotations', 'instances_val2017.json')
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Initialize COCO API
        self.coco = coco.COCO(ann_file)
        self.image_dir = image_dir

        # Get all image IDs
        self.image_ids = list(self.coco.imgs.keys())

        print(f"Loaded COCO {self.split} set: {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            dict: {
                'image': (3, H, W) tensor,
                'original_size': (H, W) tuple,
                'ground_truth_mask': (H, W) tensor,
                'prompt': dict with point_coords, point_labels, or boxes
            }
        """
        image_id = self.image_ids[idx]

        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        original_size = image.size[::-1]  # (H, W)

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        if len(annotations) == 0:
            # No annotations, create empty mask
            mask = np.zeros(original_size, dtype=np.uint8)
        else:
            # Combine all masks
            masks = []
            for ann in annotations:
                if ann['iscrowd'] == 0:
                    rle = self.coco.annToRLE(ann)
                    mask = coco_mask.decode(rle)
                    masks.append(mask)

            if len(masks) > 0:
                mask = np.stack(masks).max(axis=0)  # Combine masks
            else:
                mask = np.zeros(original_size, dtype=np.uint8)

        # Resize image and mask
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # (3, H, W)

        # Resize mask
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize((self.image_size, self.image_size), resample=PILImage.NEAREST)
        mask = np.array(mask_pil)
        mask = torch.from_numpy(mask).float() / 255.0

        # Generate prompt
        prompt = self._generate_prompt(mask)

        return {
            'image': image,
            'original_size': original_size,
            'ground_truth_mask': mask,
            'prompt': prompt,
        }

    def _generate_prompt(self, mask):
        """Generate prompt from ground truth mask."""
        if self.prompt_strategy == 'random':
            point_coords, point_labels = SAMPromptGenerator.random_point(mask, self.num_prompts)
            return {
                'point_coords': torch.from_numpy(point_coords),
                'point_labels': torch.from_numpy(point_labels),
            }
        elif self.prompt_strategy == 'center':
            point_coords, point_labels = SAMPromptGenerator.center_point(mask)
            return {
                'point_coords': torch.from_numpy(point_coords),
                'point_labels': torch.from_numpy(point_labels),
            }
        elif self.prompt_strategy == 'box':
            box = SAMPromptGenerator.bounding_box(mask)
            return {
                'boxes': torch.from_numpy(box),
            }
        elif self.prompt_strategy == 'grid':
            point_coords, point_labels = SAMPromptGenerator.grid_points(mask)
            return {
                'point_coords': torch.from_numpy(point_coords),
                'point_labels': torch.from_numpy(point_labels),
            }
        else:
            raise ValueError(f"Invalid prompt strategy: {self.prompt_strategy}")


def collate_sam_fn(batch):
    """
    Custom collate function for SAM dataset.

    SAM processes images one at a time due to varying prompt types,
    so we return them as a list without batching.
    """
    return batch


def get_sam_dataloader(data_root, split='train', dataset_name='coco',
                        batch_size=8, num_workers=4, prompt_strategy='random',
                        num_prompts=1, image_size=1024):
    """
    Create a SAM dataset loader.

    Args:
        data_root: Root directory of dataset
        split: 'train', 'val', or 'test'
        dataset_name: 'coco' or other supported dataset
        batch_size: Batch size (note: SAM processes one image at a time)
        num_workers: Number of data loading workers
        prompt_strategy: Prompt generation strategy
        num_prompts: Number of prompts per image
        image_size: Target image size

    Returns:
        DataLoader instance
    """
    dataset = SAMDataset(
        data_root=data_root,
        split=split,
        dataset_name=dataset_name,
        prompt_strategy=prompt_strategy,
        num_prompts=num_prompts,
        image_size=image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sam_fn,
        drop_last=(split == 'train'),
    )

    return dataloader
