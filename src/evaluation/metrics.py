"""
metrics.py — Segmentation metrics computed from scratch.

All metrics are computed from a confusion matrix for numerical stability
and to handle full-resolution images without OOM.

Metrics:
  - Per-class IoU (Intersection over Union): IoU_c = TP_c / (TP_c + FP_c + FN_c)
  - mIoU (mean IoU): average of per-class IoU
  - Per-class F1 (Dice coefficient): F1_c = 2 * TP_c / (2*TP_c + FP_c + FN_c)
  - Pixel accuracy: (sum TP) / total_pixels
  - Mean accuracy: average of per-class accuracies

Relationship between IoU and Dice:
  Dice = 2 * IoU / (1 + IoU)
  IoU = Dice / (2 - Dice)
  Both measure overlap; Dice gives higher values for the same overlap.

Confusion matrix accumulator:
  Implemented as a streaming accumulator to handle full-resolution images
  without loading all predictions into memory at once.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ConfusionMatrix:
    """Streaming confusion matrix accumulator.

    Accumulates TP, FP, FN per class across batches without storing
    all predictions. Can handle arbitrarily many images.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)
        self.total_pixels = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Update confusion matrix with a batch of predictions.

        Args:
            pred: (H, W) or (B, H, W) array of predicted class indices.
            target: (H, W) or (B, H, W) array of ground truth class indices.
        """
        if pred.ndim == 2:
            pred = pred[np.newaxis, :, :]
        if target.ndim == 2:
            target = target[np.newaxis, :, :]

        B, H, W = pred.shape
        self.total_pixels += B * H * W

        for c in range(self.num_classes):
            pred_c = pred == c
            target_c = target == c

            self.tp[c] += np.logical_and(pred_c, target_c).sum()
            self.fp[c] += np.logical_and(pred_c, np.logical_not(target_c)).sum()
            self.fn[c] += np.logical_and(np.logical_not(pred_c), target_c).sum()

    def compute_iou(self, exclude_background: bool = False) -> np.ndarray:
        """Compute per-class IoU.

        Args:
            exclude_background: If True, return IoU for classes 1..N-1 only.

        Returns:
            Array of per-class IoU values.
        """
        denominator = self.tp + self.fp + self.fn
        denominator = np.maximum(denominator, 1)  # Avoid division by zero
        iou = self.tp / denominator.astype(np.float64)

        if exclude_background:
            return iou[1:]
        return iou

    def compute_miou(self, exclude_background: bool = False) -> float:
        """Compute mean IoU."""
        iou = self.compute_iou(exclude_background=exclude_background)
        return float(iou.mean())

    def compute_f1(self, exclude_background: bool = False) -> np.ndarray:
        """Compute per-class F1 (Dice coefficient)."""
        denominator = 2 * self.tp + self.fp + self.fn
        denominator = np.maximum(denominator, 1)
        f1 = (2 * self.tp) / denominator.astype(np.float64)

        if exclude_background:
            return f1[1:]
        return f1

    def compute_pixel_accuracy(self) -> float:
        """Compute pixel-wise accuracy."""
        total_correct = self.tp.sum()
        return float(total_correct / max(self.total_pixels, 1))

    def compute_mean_accuracy(self, exclude_background: bool = False) -> float:
        """Compute mean of per-class accuracies."""
        denominator = self.tp + self.fn
        denominator = np.maximum(denominator, 1)
        per_class_acc = self.tp / denominator.astype(np.float64)

        if exclude_background:
            per_class_acc = per_class_acc[1:]

        return float(per_class_acc.mean())

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the full confusion matrix (num_classes × num_classes)."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        # This requires storing all predictions which defeats the streaming purpose.
        # Use this only when you have the full pred and target arrays.
        raise NotImplementedError(
            "Use compute_confusion_matrix(pred, target, num_classes) instead"
        )


def compute_confusion_matrix(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute full confusion matrix from prediction and target arrays.

    Args:
        pred: (H, W) predicted class indices.
        target: (H, W) ground truth class indices.
        num_classes: Number of classes.

    Returns:
        (num_classes, num_classes) confusion matrix where cm[i, j] = count
        of pixels with true class i predicted as class j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = np.stack([target.ravel(), pred.ravel()], axis=0)
    idx = np.ravel_multi_index(idx, (num_classes, num_classes))
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    return cm.astype(np.int64)


def compute_all_metrics(
    cm: ConfusionMatrix, class_names: List[str], exclude_background: bool = False
) -> Dict:
    """Compute all metrics from a ConfusionMatrix accumulator.

    Args:
        cm: Populated ConfusionMatrix.
        class_names: List of class name strings.
        exclude_background: Whether to exclude background from mIoU.

    Returns:
        Dictionary with all metrics.
    """
    iou = cm.compute_iou(exclude_background=False)
    f1 = cm.compute_f1(exclude_background=False)
    pixel_acc = cm.compute_pixel_accuracy()
    mean_acc = cm.compute_mean_accuracy(exclude_background=False)
    miou = cm.compute_miou(exclude_background=exclude_background)

    results = {
        "miou": miou,
        "pixel_accuracy": pixel_acc,
        "mean_accuracy": mean_acc,
    }

    for i, name in enumerate(class_names):
        results[f"{name}/iou"] = float(iou[i])
        results[f"{name}/f1"] = float(f1[i])

    return results
