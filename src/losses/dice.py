"""
dice.py — Multiclass Dice loss implemented from scratch.

Dice coefficient:
    Dice = 2 * |A ∩ B| / (|A| + |B|)
         = 2 * Σ(p * y) / (Σp + Σy)

Dice loss:
    DiceLoss = 1 - Dice

For multiclass segmentation, we compute the Dice coefficient per-class
and average across classes. The smooth term (ε) ensures numerical stability
when both prediction and target are empty for a class.

Why Dice loss for segmentation:
  1. Directly optimises overlap (related to IoU/metrics)
  2. Handles class imbalance naturally (each class contributes equally)
  3. Complementary to cross-entropy: CE optimises per-pixel accuracy while
     Dice optimises region overlap

Background class handling:
  The background class (index 0) in LoveDA is often very large and smooth.
  Including it in the Dice loss can dominate the gradient. We optionally
  exclude it via the exclude_background flag.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Multiclass Dice loss.

    Args:
        smooth: Smoothing factor for numerical stability (default: 1e-6).
        exclude_background: If True, exclude class 0 from loss (default: False).
        include_background_in_mean: If True, include background in the class average.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        exclude_background: bool = False,
    ):
        super().__init__()
        self.smooth = smooth
        self.exclude_background = exclude_background

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) raw logits.
            target: (B, H, W) integer class indices.

        Returns:
            Scalar Dice loss.
        """
        # Convert logits to probabilities
        probs = torch.softmax(pred, dim=1)

        # One-hot encode target
        B, C, H, W = pred.shape
        target_one_hot = torch.zeros_like(probs)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)

        # Compute Dice per class
        dims = (0, 2, 3)  # Sum over batch and spatial dimensions
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        cardinality = torch.sum(probs + target_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice

        # Exclude background if requested
        if self.exclude_background:
            dice_loss = dice_loss[1:]

        return dice_loss.mean()
