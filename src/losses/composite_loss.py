"""
composite_loss.py — BoundaryAwareLoss combining CE + Dice + Boundary components.

The final loss function used by the custom model:
    L = α * CE + β * Dice + γ * Boundary

Default values (chosen after validation experiments):
    α = 1.0  — Cross-entropy weight (keep at 1.0 as reference)
    β = 1.0  — Dice weight (equal contribution with CE)
    γ = 0.5  — Boundary weight (auxiliary, so lower weight)

Why these defaults:
  - CE and Dice are complementary: CE is per-pixel, Dice is region-based
  - Equal weighting (1.0 each) gives balanced optimisation of both
  - Boundary loss at 0.5 provides focused boundary refinement without
    dominating the total loss
  - Total loss magnitude ≈ 2.5 at init, which is reasonable for AdamW

Deep supervision support:
  When the model has auxiliary heads, each head contributes to the loss:
    Total = L_main + Σ(w_i * L_aux_i)
  where w_i are annealed from their initial values to 0 over the first
  aux_weight_anneal_epochs.

Class weights are loaded from config and passed to WeightedCrossEntropyLoss.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.losses.cross_entropy import WeightedCrossEntropyLoss
from src.losses.dice import DiceLoss
from src.losses.boundary_loss import BoundaryLoss


class BoundaryAwareLoss(nn.Module):
    """
    Composite boundary-aware loss: L = α * CE + β * Dice + γ * Boundary.

    Args:
        config: Configuration dictionary (loss section).
        class_weights: Per-class weights for weighted CE.
        num_classes: Number of segmentation classes.
    """

    def __init__(
        self,
        config: Dict,
        class_weights: Optional[torch.Tensor] = None,
        num_classes: int = 7,
    ):
        super().__init__()
        loss_cfg = config.get("loss", {})

        self.alpha = loss_cfg.get("alpha", 1.0)
        self.beta = loss_cfg.get("beta", 1.0)
        self.gamma = loss_cfg.get("gamma", 0.5)

        # Cross-entropy
        self.ce_loss = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
        )

        # Dice loss
        self.dice_loss = DiceLoss(
            smooth=loss_cfg.get("dice_smooth", 1e-6),
        )

        # Boundary-aware loss
        boundary_cfg = loss_cfg.get("boundary", {})
        self.use_boundary = boundary_cfg.get("enabled", True) and self.gamma > 0
        if self.use_boundary:
            self.boundary_loss = BoundaryLoss(
                kernel_size=boundary_cfg.get("kernel_size", 3),
                boundary_width=boundary_cfg.get("boundary_width", 3),
                boundary_weight=boundary_cfg.get("boundary_weight", 2.0),
                num_classes=num_classes,
            )

        # Deep supervision
        self.deep_supervision_weights = config["model"].get(
            "deep_supervision_weights", [1.0, 0.5, 0.25]
        )
        self.aux_weight_anneal_epochs = config["model"].get(
            "aux_weight_anneal_epochs", 10
        )

    def get_aux_weights(self, current_epoch: int) -> List[float]:
        """Compute annealed auxiliary loss weights."""
        if current_epoch >= self.aux_weight_anneal_epochs:
            return [0.0] * (len(self.deep_supervision_weights) - 1)
        progress = current_epoch / max(self.aux_weight_anneal_epochs, 1)
        aux_weights = [
            w * (1.0 - progress) for w in self.deep_supervision_weights[1:]
        ]
        return aux_weights

    def compute_base_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the composite loss (CE + Dice + Boundary) for a single prediction."""
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        total = self.alpha * ce + self.beta * dice

        if self.use_boundary:
            # Per-pixel CE loss for boundary weighting
            ce_per_pixel = torch.nn.functional.cross_entropy(
                pred, target, reduction="none"
            )
            boundary = self.boundary_loss(pred, target, ce_per_pixel)
            total = total + self.gamma * boundary

        return total, {"ce": ce.item(), "dice": dice.item(), "total": total.item()}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            pred: Either a single tensor (B, C, H, W) or (main_pred, [aux_preds]).
            target: (B, H, W) integer class indices.
            current_epoch: Current training epoch (for aux weight annealing).

        Returns:
            Scalar loss.
        """
        aux_weights = self.get_aux_weights(current_epoch)

        if isinstance(pred, (list, tuple)):
            main_pred = pred[0]
            aux_preds = pred[1]
        else:
            main_pred = pred
            aux_preds = []

        # Main loss
        main_loss, main_log = self.compute_base_loss(main_pred, target)

        # Auxiliary losses (deep supervision)
        aux_loss = 0.0
        for i, aux_pred in enumerate(aux_preds):
            if i < len(aux_weights) and aux_weights[i] > 0:
                # Resize target to match aux prediction resolution if needed
                aux_target = target
                if aux_pred.shape[2:] != target.shape[2:]:
                    aux_target = torch.nn.functional.interpolate(
                        target.unsqueeze(1).float(),
                        size=aux_pred.shape[2:],
                        mode="nearest",
                    ).squeeze(1).long()

                loss_aux, _ = self.compute_base_loss(aux_pred, aux_target)
                aux_loss = aux_loss + aux_weights[i] * loss_aux

        return main_loss + aux_loss
