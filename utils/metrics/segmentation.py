"""
Segmentation metrics for crack detection.
"""

import numpy as np
import torch
from typing import Dict, Union, Tuple


class SegmentationMetrics:
    """
    Accumulator for segmentation metrics.

    Computes (following MTSCrack paper eq. 27-32):
        - Precision (P)
        - Recall (R)
        - F1-Score
        - Dice coefficient
        - IoU  — crack-class IoU
        - MIoU — mean of crack-class and background IoU
        - PA   — overall pixel accuracy = (TP+TN)/total  (eq. 32)
        - MPA  — mean per-class recall = (R_crack + R_bg)/2  (eq. 31)
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        """
        Initialize metrics accumulator.

        Args:
            threshold: Threshold for binarizing predictions
            eps: Small value to avoid division by zero
        """
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated values."""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        self.tn = 0  # True negatives

        # Per-image metrics for std calculation
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.dices = []
        self.ious = []
        self.mious = []
        self.pas = []    # Pixel Accuracy
        self.mpas = []   # Mean Pixel Accuracy

    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> None:
        """
        Update metrics with batch predictions.

        Args:
            preds: Predictions (B, 1, H, W) or (B, H, W), values in [0, 1] or binary
            targets: Ground truth (B, 1, H, W) or (B, H, W), binary
        """
        # Convert to numpy
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Ensure 4D: (B, 1, H, W)
        if preds.ndim == 3:
            preds = preds[:, np.newaxis, :, :]
        if targets.ndim == 3:
            targets = targets[:, np.newaxis, :, :]

        # Binarize predictions if not already
        if preds.max() > 1 or (preds.min() >= 0 and preds.max() <= 1 and not np.allclose(preds, preds.astype(int))):
            preds = (preds > self.threshold).astype(np.float32)

        # Binarize targets
        targets = (targets > 0.5).astype(np.float32)

        batch_size = preds.shape[0]

        for i in range(batch_size):
            pred = preds[i].flatten()
            target = targets[i].flatten()

            # Compute confusion matrix elements
            tp = np.sum((pred == 1) & (target == 1))
            fp = np.sum((pred == 1) & (target == 0))
            fn = np.sum((pred == 0) & (target == 1))
            tn = np.sum((pred == 0) & (target == 0))

            # Accumulate global
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn

            # Compute per-image metrics
            precision = tp / (tp + fp + self.eps)
            recall = tp / (tp + fn + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            dice = 2 * tp / (2 * tp + fp + fn + self.eps)
            iou_crack = tp / (tp + fp + fn + self.eps)
            iou_bg = tn / (tn + fp + fn + self.eps)
            miou = (iou_crack + iou_bg) / 2

            # PA: overall pixel accuracy = (TP + TN) / all_pixels  (eq. 32)
            total_pixels = tp + fp + fn + tn
            pa = (tp + tn) / (total_pixels + self.eps)

            # MPA: mean per-class recall = (recall_crack + recall_bg) / 2  (eq. 31)
            recall_bg = tn / (tn + fp + self.eps)
            mpa = (recall + recall_bg) / 2

            self.precisions.append(precision)
            self.recalls.append(recall)
            self.f1s.append(f1)
            self.dices.append(dice)
            self.ious.append(iou_crack)  # IoU of crack class (main metric)
            self.mious.append(miou)
            self.pas.append(pa)
            self.mpas.append(mpa)

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary of metrics with mean values
        """
        # Global metrics
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + self.eps)
        iou_crack = self.tp / (self.tp + self.fp + self.fn + self.eps)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + self.eps)
        miou = (iou_crack + iou_bg) / 2

        # PA: overall pixel accuracy = (TP + TN) / all_pixels  (eq. 32)
        total = self.tp + self.fp + self.fn + self.tn
        pa = (self.tp + self.tn) / (total + self.eps)

        # MPA: mean per-class recall = (recall_crack + recall_bg) / 2  (eq. 31)
        recall_bg = self.tn / (self.tn + self.fp + self.eps)
        mpa = (recall + recall_bg) / 2

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'dice': float(dice),
            'iou': float(iou_crack),
            'miou': float(miou),
            'pa': float(pa),
            'mpa': float(mpa),
        }

    def compute_with_std(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute metrics with standard deviation.

        Returns:
            Dictionary of metrics with (mean, std) tuples
        """
        return {
            'precision': (np.mean(self.precisions), np.std(self.precisions)),
            'recall': (np.mean(self.recalls), np.std(self.recalls)),
            'f1': (np.mean(self.f1s), np.std(self.f1s)),
            'dice': (np.mean(self.dices), np.std(self.dices)),
            'iou': (np.mean(self.ious), np.std(self.ious)),
            'miou': (np.mean(self.mious), np.std(self.mious)),
            'pa': (np.mean(self.pas), np.std(self.pas)),
            'mpa': (np.mean(self.mpas), np.std(self.mpas)),
        }


def compute_metrics(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all metrics for a batch.

    Args:
        preds: Predictions
        targets: Ground truth
        threshold: Binarization threshold

    Returns:
        Dictionary of metrics
    """
    metrics = SegmentationMetrics(threshold=threshold)
    metrics.update(preds, targets)
    return metrics.compute()


def compute_pr_curve(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_thresholds: int = 101
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve with proper interpolation.

    The PR curve is computed by varying the threshold and computing precision
    and recall at each point. The curve is then interpolated to ensure precision
    is monotonically decreasing as recall increases (standard PR curve convention).

    Args:
        preds: Predictions (probabilities)
        targets: Ground truth (binary)
        num_thresholds: Number of threshold points

    Returns:
        Tuple of (precisions, recalls, thresholds) - sorted by recall ascending,
        with precision monotonically non-increasing
    """
    # Convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds = preds.flatten()
    targets = (targets.flatten() > 0.5).astype(np.float32)

    # Use thresholds from high to low (1 to 0)
    # This naturally gives recall from low to high
    thresholds = np.linspace(1, 0, num_thresholds)
    precisions = []
    recalls = []
    eps = 1e-7

    for thresh in thresholds:
        pred_binary = (preds >= thresh).astype(np.float32)

        tp = np.sum((pred_binary == 1) & (targets == 1))
        fp = np.sum((pred_binary == 1) & (targets == 0))
        fn = np.sum((pred_binary == 0) & (targets == 1))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort by recall (ascending)
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    thresholds = thresholds[sorted_indices]

    # Apply interpolation: make precision monotonically decreasing
    # For each recall value, precision should be the maximum precision
    # at any recall >= current recall (right-to-left max)
    precisions_interp = np.zeros_like(precisions)
    precisions_interp[-1] = precisions[-1]
    for i in range(len(precisions) - 2, -1, -1):
        precisions_interp[i] = max(precisions[i], precisions_interp[i + 1])

    return precisions_interp, recalls, thresholds


def compute_average_precision(
    precisions: np.ndarray,
    recalls: np.ndarray
) -> float:
    """
    Compute Average Precision (AP) from PR curve.

    Uses the interpolated precision at 11 recall levels (PASCAL VOC style)
    or trapezoidal integration if already interpolated.

    Args:
        precisions: Precision values (should already be sorted by recall)
        recalls: Recall values (should already be sorted ascending)

    Returns:
        Average precision score
    """
    # Ensure arrays are sorted by recall (ascending)
    if len(recalls) > 1 and recalls[0] > recalls[-1]:
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]

    # Add sentinel values at boundaries
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    # Ensure precision is monotonically decreasing (max from right)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AP using trapezoidal rule
    ap = np.trapezoid(precisions, recalls)

    return float(np.clip(ap, 0, 1))
