"""
cross_entropy.py — Weighted cross-entropy loss with label smoothing.

Plain cross-entropy struggles with class imbalance in aerial imagery:
  - Background and Agriculture classes dominate (~60% of pixels combined)
  - Building, Water, and Barren are rare
  - The model can achieve low loss by predicting majority classes everywhere

Weighted cross-entropy addresses this by up-weighting rare classes:
    CE = - Σ w_c * y_c * log(p_c)

Label smoothing (ε = 0.1) replaces hard targets with soft targets:
    y'_c = (1 - ε) * y_c + ε / K
  This prevents the model from becoming overconfident, improves
  generalisation, and is particularly helpful for noisy boundary regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted cross-entropy loss with optional label smoothing.

    Args:
        class_weights: Tensor of shape (num_classes,) with per-class weights.
                       Can be loaded from dataset_stats.yaml.
        label_smoothing: Epsilon for label smoothing (default 0.0 = none).
        ignore_index: Class index to ignore in loss (default: -1 = none).
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) raw logits.
            target: (B, H, W) integer class indices.

        Returns:
            Scalar loss.
        """
        B, C, H, W = pred.shape

        if self.label_smoothing > 0.0:
            # Convert targets to smoothed one-hot
            with torch.no_grad():
                target_one_hot = torch.zeros_like(pred)
                target_one_hot.scatter_(
                    1, target.unsqueeze(1), 1.0
                )
                smooth_eps = self.label_smoothing / C
                target_smooth = (
                    target_one_hot * (1.0 - self.label_smoothing) + smooth_eps
                )

            # Log-softmax predictions
            log_probs = F.log_softmax(pred, dim=1)

            # Weighted cross-entropy with smoothed targets
            loss = -target_smooth * log_probs

            if self.class_weights is not None:
                weights = self.class_weights.to(pred.device).view(1, C, 1, 1)
                loss = loss * weights

            loss = loss.sum(dim=1)

            if self.ignore_index >= 0:
                mask = target != self.ignore_index
                loss = loss[mask]
                if loss.numel() == 0:
                    return torch.tensor(0.0, device=pred.device)
                return loss.mean()

            return loss.mean()

        # Standard weighted cross-entropy (no label smoothing)
        loss = F.cross_entropy(
            pred,
            target,
            weight=self.class_weights.to(pred.device) if self.class_weights is not None else None,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else -100,
        )
        return loss
