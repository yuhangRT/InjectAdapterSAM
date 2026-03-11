"""
Segmentation metrics for SAM evaluation.

This module provides common segmentation metrics including IoU, Dice,
precision, recall, and F1 score for evaluating SAM with adapter.
"""

import torch
import numpy as np

__all__ = ['compute_iou', 'compute_dice', 'compute_mask_iou',
           'SAMMetrics', 'compute_precision_recall_f1']


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU) for binary masks.

    Args:
        pred_mask: (N, H, W) predicted binary masks
        gt_mask: (N, H, W) ground truth binary masks

    Returns:
        (N,) tensor of IoU scores
    """
    assert pred_mask.shape == gt_mask.shape, "Shape mismatch"

    # Flatten
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)

    # Compute intersection and union
    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection

    # Handle empty masks
    iou = intersection / (union + 1e-8)

    return iou


def compute_dice(pred_mask, gt_mask):
    """
    Compute Dice coefficient for binary masks.

    Args:
        pred_mask: (N, H, W) predicted binary masks
        gt_mask: (N, H, W) ground truth binary masks

    Returns:
        (N,) tensor of Dice scores
    """
    assert pred_mask.shape == gt_mask.shape, "Shape mismatch"

    # Flatten
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)

    # Compute Dice
    intersection = (pred_flat * gt_flat).sum(dim=1)
    dice = (2 * intersection) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1e-8)

    return dice


def compute_mask_iou(pred_masks, gt_masks, batch=True):
    """
    Batch-wise IoU computation handling multiple masks per image.

    Args:
        pred_masks: (N, K, H, W) predicted masks (K masks per image)
        gt_masks: (N, H, W) ground truth masks
        batch: If True, average over masks per image

    Returns:
        If batch=True: (N,) tensor of mean IoU per image
        If batch=False: (N, K) tensor of IoU for each mask
    """
    if pred_masks.dim() == 3:
        # Add mask dimension if missing
        pred_masks = pred_masks.unsqueeze(1)

    N, K, H, W = pred_masks.shape

    # Flatten spatial dimensions
    pred_flat = pred_masks.view(N, K, -1)
    gt_flat = gt_masks.view(N, -1).unsqueeze(1).expand(N, K, -1)

    # Compute IoU for each mask
    intersection = (pred_flat * gt_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + gt_flat.sum(dim=2) - intersection

    iou = intersection / (union + 1e-8)  # (N, K)

    if batch:
        return iou.mean(dim=1)  # (N,)
    return iou


def compute_precision_recall_f1(pred_mask, gt_mask):
    """
    Compute precision, recall, and F1 score for binary masks.

    Args:
        pred_mask: (N, H, W) predicted binary masks
        gt_mask: (N, H, W) ground truth binary masks

    Returns:
        dict with 'precision', 'recall', 'f1' as (N,) tensors
    """
    assert pred_mask.shape == gt_mask.shape, "Shape mismatch"

    # Flatten
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)

    # True positives, false positives, false negatives
    tp = (pred_flat * gt_flat).sum(dim=1)
    fp = pred_flat.sum(dim=1) - tp
    fn = gt_flat.sum(dim=1) - tp

    # Compute metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class SAMMetrics:
    """
    Aggregator for SAM evaluation metrics.

    Tracks running averages of IoU, Dice, Precision, Recall, and F1.

    Example:
        >>> metrics = SAMMetrics()
        >>> for batch in dataloader:
        ...     preds = model(batch)
        ...     metrics.update(preds, batch['ground_truth_mask'])
        >>> results = metrics.compute()
        >>> print(f"Mean IoU: {results['iou']:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.f1_sum = 0.0
        self.count = 0

    def update(self, pred_masks, gt_masks):
        """
        Update metrics with a new batch of predictions.

        Args:
            pred_masks: (N, H, W) predicted binary masks
            gt_masks: (N, H, W) ground truth binary masks
        """
        # Convert to binary if needed
        if pred_masks.dtype == torch.float32:
            pred_masks = (pred_masks > 0.5).float()

        if gt_masks.dtype == torch.float32:
            gt_masks = (gt_masks > 0.5).float()

        # Compute metrics
        iou = compute_iou(pred_masks, gt_masks)
        dice = compute_dice(pred_masks, gt_masks)
        prf = compute_precision_recall_f1(pred_masks, gt_masks)

        # Update sums
        self.iou_sum += iou.sum().item()
        self.dice_sum += dice.sum().item()
        self.precision_sum += prf['precision'].sum().item()
        self.recall_sum += prf['recall'].sum().item()
        self.f1_sum += prf['f1'].sum().item()
        self.count += pred_masks.size(0)

    def compute(self):
        """
        Compute aggregated metrics.

        Returns:
            dict with 'iou', 'dice', 'precision', 'recall', 'f1' as floats
        """
        if self.count == 0:
            return {
                'iou': 0.0,
                'dice': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }

        return {
            'iou': self.iou_sum / self.count,
            'dice': self.dice_sum / self.count,
            'precision': self.precision_sum / self.count,
            'recall': self.recall_sum / self.count,
            'f1': self.f1_sum / self.count,
        }

    def __str__(self):
        """String representation of current metrics."""
        metrics = self.compute()
        return (f"IoU: {metrics['iou']:.4f}, "
                f"Dice: {metrics['dice']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}")


def compute_mean_iou(pred_masks, gt_masks):
    """
    Compute mean IoU over a batch of masks.

    Args:
        pred_masks: List of (H, W) predicted masks
        gt_masks: List of (H, W) ground truth masks

    Returns:
        float: Mean IoU over all masks
    """
    total_iou = 0.0
    valid_count = 0

    for pred, gt in zip(pred_masks, gt_masks):
        # Convert to tensors if needed
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred)
        if not isinstance(gt, torch.Tensor):
            gt = torch.from_numpy(gt)

        # Add batch dimension if needed
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)

        # Compute IoU
        iou = compute_iou(pred.float(), gt.float())
        total_iou += iou.item()
        valid_count += 1

    return total_iou / max(valid_count, 1)


def compute_panoptic_quality(pred_masks, gt_masks, iou_threshold=0.5):
    """
    Compute Panoptic Quality metric.

    PQ = SQ * DQ where:
    - SQ (Segmentation Quality): mean IoU of matched segments
    - DQ (Detection Quality): F1 score of matched segments

    Args:
        pred_masks: List of predicted instance masks
        gt_masks: List of ground truth instance masks
        iou_threshold: IoU threshold for matching

    Returns:
        float: Panoptic Quality score
    """
    # This is a simplified version - full PQ requires instance IDs
    # For now, we compute mean IoU as a proxy

    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0

    total_iou = 0.0
    matches = 0

    for pred in pred_masks:
        best_iou = 0.0
        for gt in gt_masks:
            iou = compute_iou(
                pred.unsqueeze(0).float(),
                gt.unsqueeze(0).float()
            ).item()
            best_iou = max(best_iou, iou)

        if best_iou >= iou_threshold:
            total_iou += best_iou
            matches += 1

    if matches == 0:
        return 0.0

    sq = total_iou / matches
    dq = 2 * matches / (len(pred_masks) + len(gt_masks))

    return sq * dq
