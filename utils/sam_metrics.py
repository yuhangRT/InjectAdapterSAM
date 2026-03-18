"""
Segmentation metrics for WireCR-SAM semantic segmentation.
"""

from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "compute_iou",
    "compute_dice",
    "compute_precision_recall_f1",
    "compute_boundary_f1",
    "compute_cldice",
    "SAMMetrics",
]


EPS = 1e-8


def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.float().view(pred_mask.size(0), -1)
    gt_mask = gt_mask.float().view(gt_mask.size(0), -1)
    intersection = (pred_mask * gt_mask).sum(dim=1)
    union = pred_mask.sum(dim=1) + gt_mask.sum(dim=1) - intersection
    return intersection / (union + EPS)


def compute_dice(pred_mask, gt_mask):
    pred_mask = pred_mask.float().view(pred_mask.size(0), -1)
    gt_mask = gt_mask.float().view(gt_mask.size(0), -1)
    intersection = (pred_mask * gt_mask).sum(dim=1)
    return (2 * intersection) / (pred_mask.sum(dim=1) + gt_mask.sum(dim=1) + EPS)


def compute_precision_recall_f1(pred_mask, gt_mask):
    pred_mask = pred_mask.float().view(pred_mask.size(0), -1)
    gt_mask = gt_mask.float().view(gt_mask.size(0), -1)

    tp = (pred_mask * gt_mask).sum(dim=1)
    fp = pred_mask.sum(dim=1) - tp
    fn = gt_mask.sum(dim=1) - tp

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    return {"precision": precision, "recall": recall, "f1": f1}


def _binary_boundary_map(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded > 0).float()


def compute_boundary_f1(pred_mask, gt_mask):
    pred_boundary = _binary_boundary_map(pred_mask)
    gt_boundary = _binary_boundary_map(gt_mask)
    prf = compute_precision_recall_f1(pred_boundary.squeeze(1), gt_boundary.squeeze(1))
    return prf["f1"]


def _soft_erode(img):
    if img.dim() == 3:
        img = img.unsqueeze(1)
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    if img.dim() == 3:
        img = img.unsqueeze(1)
    return F.max_pool2d(img, 3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def _soft_skeletonize(img, iterations=10):
    if img.dim() == 3:
        img = img.unsqueeze(1)
    img = img.float()
    opened = _soft_open(img)
    skeleton = F.relu(img - opened)
    for _ in range(iterations - 1):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def compute_cldice(pred_mask, gt_mask, iterations=10):
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)

    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    skeleton_pred = _soft_skeletonize(pred_mask, iterations=iterations)
    skeleton_gt = _soft_skeletonize(gt_mask, iterations=iterations)

    tprec = (skeleton_pred * gt_mask).sum(dim=(1, 2, 3)) / (skeleton_pred.sum(dim=(1, 2, 3)) + EPS)
    tsens = (skeleton_gt * pred_mask).sum(dim=(1, 2, 3)) / (skeleton_gt.sum(dim=(1, 2, 3)) + EPS)

    return 2 * tprec * tsens / (tprec + tsens + EPS)


class SAMMetrics:
    """Aggregates semantic segmentation metrics for WireCR-SAM."""

    def __init__(self, num_classes=3, class_names: Optional[Sequence[str]] = None):
        self.num_classes = num_classes
        if class_names is None:
            if num_classes == 2:
                class_names = ["background", "foreground"]
            elif num_classes == 3:
                class_names = ["background", "wire", "interface-hole"]
            else:
                class_names = ["background"] + [f"class_{idx}" for idx in range(1, num_classes)]
        self.class_names = list(class_names[:num_classes])
        if len(self.class_names) < num_classes:
            self.class_names += [f"class_{idx}" for idx in range(len(self.class_names), num_classes)]
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)
        self.boundary_f1_sum = 0.0
        self.cldice_sum = 0.0
        self.sample_count = 0

    def update(self, pred_labels, gt_labels):
        pred_labels = pred_labels.detach().long().cpu()
        gt_labels = gt_labels.detach().long().cpu()

        if pred_labels.shape != gt_labels.shape:
            raise ValueError(f"Shape mismatch: {pred_labels.shape} vs {gt_labels.shape}")

        flat_pred = pred_labels.view(-1)
        flat_gt = gt_labels.view(-1)
        mask = (flat_gt >= 0) & (flat_gt < self.num_classes)
        encoded = self.num_classes * flat_gt[mask] + flat_pred[mask]
        bincount = torch.bincount(encoded, minlength=self.num_classes ** 2).double()
        self.confusion += bincount.view(self.num_classes, self.num_classes)

        pred_fg = (pred_labels > 0).float()
        gt_fg = (gt_labels > 0).float()
        self.boundary_f1_sum += compute_boundary_f1(pred_fg, gt_fg).sum().item()

        if self.num_classes > 1:
            pred_wire = (pred_labels == 1).float()
            gt_wire = (gt_labels == 1).float()
            self.cldice_sum += compute_cldice(pred_wire, gt_wire).sum().item()

        self.sample_count += pred_labels.shape[0]

    def compute(self):
        true_positive = torch.diag(self.confusion)
        false_positive = self.confusion.sum(dim=0) - true_positive
        false_negative = self.confusion.sum(dim=1) - true_positive
        union = true_positive + false_positive + false_negative

        class_iou = true_positive / (union + EPS)
        class_dice = (2 * true_positive) / (2 * true_positive + false_positive + false_negative + EPS)
        class_precision = true_positive / (true_positive + false_positive + EPS)
        class_recall = true_positive / (true_positive + false_negative + EPS)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + EPS)

        results = {
            "iou": class_iou.mean().item(),
            "dice": class_dice.mean().item(),
            "precision": class_precision.mean().item(),
            "recall": class_recall.mean().item(),
            "f1": class_f1.mean().item(),
            "boundary_f1": self.boundary_f1_sum / max(self.sample_count, 1),
            "cldice": self.cldice_sum / max(self.sample_count, 1),
            "class_iou": {},
            "class_dice": {},
            "class_precision": {},
            "class_recall": {},
            "class_f1": {},
        }

        for idx, name in enumerate(self.class_names):
            results["class_iou"][name] = class_iou[idx].item()
            results["class_dice"][name] = class_dice[idx].item()
            results["class_precision"][name] = class_precision[idx].item()
            results["class_recall"][name] = class_recall[idx].item()
            results["class_f1"][name] = class_f1[idx].item()

        if "wire" in results["class_iou"]:
            results["wire_iou"] = results["class_iou"]["wire"]
            results["wire_dice"] = results["class_dice"]["wire"]
        hole_key = None
        for candidate in ("interface-hole", "hole"):
            if candidate in results["class_iou"]:
                hole_key = candidate
                break
        if hole_key is not None:
            results["hole_iou"] = results["class_iou"][hole_key]
            results["hole_recall"] = results["class_recall"][hole_key]
            results["interface_hole_iou"] = results["class_iou"][hole_key]
            results["interface_hole_recall"] = results["class_recall"][hole_key]
        if "foreground" in results["class_iou"]:
            results["foreground_iou"] = results["class_iou"]["foreground"]
            results["foreground_dice"] = results["class_dice"]["foreground"]

        return results

    def __str__(self):
        metrics = self.compute()
        return (
            f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, "
            f"BoundaryF1: {metrics['boundary_f1']:.4f}, clDice: {metrics['cldice']:.4f}"
        )
