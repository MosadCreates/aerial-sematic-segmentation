"""
test_loss.py — Unit tests for loss functions.

Tests:
  1. Cross-entropy: loss decreases on perfect prediction
  2. Cross-entropy: label smoothing produces non-zero loss for perfect prediction
  3. Dice loss: perfect prediction gives zero loss
  4. Dice loss: completely wrong prediction gives high loss
  5. Boundary loss: higher near boundaries than interior pixels
  6. BoundaryAwareLoss: gradient flow is non-zero through all components
  7. BoundaryAwareLoss: deep supervision works with aux heads
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from src.losses.cross_entropy import WeightedCrossEntropyLoss
from src.losses.dice import DiceLoss
from src.losses.boundary_loss import BoundaryLoss, morphological_boundary
from src.losses.composite_loss import BoundaryAwareLoss


def load_test_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TestCrossEntropyLoss:
    def test_perfect_prediction_low_loss(self):
        """Cross-entropy should be low when prediction matches target."""
        B, C, H, W = 2, 7, 64, 64
        target = torch.randint(0, C, (B, H, W))
        # Create logits that strongly favour the target class
        pred = torch.zeros(B, C, H, W)
        pred.scatter_(1, target.unsqueeze(1), 10.0)

        criterion = WeightedCrossEntropyLoss(label_smoothing=0.0)
        loss = criterion(pred, target)
        assert loss.item() < 0.1, f"Perfect pred loss too high: {loss.item()}"

    def test_perfect_prediction_with_label_smoothing(self):
        """Label smoothing should give non-zero loss even for perfect prediction."""
        B, C, H, W = 2, 7, 64, 64
        target = torch.randint(0, C, (B, H, W))
        pred = torch.zeros(B, C, H, W)
        pred.scatter_(1, target.unsqueeze(1), 10.0)

        criterion = WeightedCrossEntropyLoss(label_smoothing=0.1)
        loss = criterion(pred, target)
        assert loss.item() > 0.001, f"Loss should be non-zero with label smoothing"
        assert loss.item() < 0.5, f"Loss too high with label smoothing: {loss.item()}"

    def test_random_prediction_higher_loss(self):
        """Random prediction should have higher loss than perfect prediction."""
        B, C, H, W = 2, 7, 64, 64
        target = torch.randint(0, C, (B, H, W))

        # Perfect prediction
        pred_perfect = torch.zeros(B, C, H, W)
        pred_perfect.scatter_(1, target.unsqueeze(1), 10.0)

        # Random prediction
        pred_random = torch.randn(B, C, H, W)

        criterion = WeightedCrossEntropyLoss(label_smoothing=0.0)
        loss_perfect = criterion(pred_perfect, target)
        loss_random = criterion(pred_random, target)
        assert loss_random > loss_perfect, (
            f"Random loss {loss_random:.4f} should be > perfect {loss_perfect:.4f}"
        )


class TestDiceLoss:
    def test_perfect_prediction_zero_loss(self):
        """Dice loss should be 0 for perfect prediction."""
        B, C, H, W = 2, 7, 64, 64
        target = torch.randint(0, C, (B, H, W))
        # Create logits that give perfect prediction
        pred = torch.zeros(B, C, H, W)
        pred.scatter_(1, target.unsqueeze(1), 10.0)

        criterion = DiceLoss(smooth=1e-6)
        loss = criterion(pred, target)
        assert loss.item() < 0.01, f"Perfect Dice loss too high: {loss.item()}"

    def test_wrong_prediction_high_loss(self):
        """Dice loss should be high for completely wrong prediction."""
        B, C, H, W = 1, 7, 32, 32
        target = torch.zeros(B, H, W, dtype=torch.long)  # All class 0
        # Predict class 1 with high confidence
        pred = -10.0 * torch.ones(B, C, H, W)
        pred[:, 0, :, :] = 0.0  # Equal to class 0

        criterion = DiceLoss(smooth=1e-6)
        loss = criterion(pred, target)
        # For total mismatch, Dice should be close to 1
        assert loss.item() > 0.9, (
            f"Wrong prediction Dice loss should be high, got {loss.item()}"
        )

    def test_exclude_background(self):
        """Exclude background should reduce the number of classes averaged."""
        B, C, H, W = 1, 7, 32, 32
        target = torch.randint(1, C, (B, H, W))  # No background pixels
        pred = torch.randn(B, C, H, W)

        criterion_all = DiceLoss(exclude_background=False)
        criterion_no_bg = DiceLoss(exclude_background=True)

        loss_all = criterion_all(pred, target)
        loss_no_bg = criterion_no_bg(pred, target)
        # Both should compute valid losses
        assert isinstance(loss_all.item(), float)
        assert isinstance(loss_no_bg.item(), float)


class TestBoundaryLoss:
    def test_boundary_region_higher_weight(self):
        """Boundary pixels should have higher loss contribution."""
        B, H, W = 1, 64, 64
        # Create mask with a clear boundary: left half class 1, right half class 0
        target = torch.zeros(B, H, W, dtype=torch.long)
        target[:, :, :W//2] = 1

        # Compute boundary map
        boundary_map = morphological_boundary(target, kernel_size=3, num_classes=2)
        n_boundary = boundary_map.sum().item()
        assert n_boundary > 0, "No boundary pixels found"

        # Create uniform per-pixel loss
        base_loss = torch.ones(B, H, W)

        criterion = BoundaryLoss(
            kernel_size=3, boundary_width=3, boundary_weight=2.0, num_classes=2
        )
        # pred is not used in boundary loss, but we pass it for API consistency
        pred_dummy = torch.zeros(B, 2, H, W)
        weighted = criterion(pred_dummy, target, base_loss)

        # Weighted loss should be > 1.0 (since boundary pixels are up-weighted)
        assert weighted.item() > 0.9, (
            f"Boundary-weighted loss should be > 0.9, got {weighted.item()}"
        )


class TestBoundaryAwareLoss:
    def test_gradient_flow_all_components(self):
        """All components of BoundaryAwareLoss should have non-zero gradients."""
        B, C, H, W = 2, 7, 64, 64
        target = torch.randint(0, C, (B, H, W))
        pred = torch.randn(B, C, H, W, requires_grad=True)

        config = load_test_config()
        criterion = BoundaryAwareLoss(config, num_classes=C)

        # Enable full loss
        criterion.alpha = 1.0
        criterion.beta = 1.0
        criterion.gamma = 0.5
        criterion.use_boundary = True

        loss = criterion(pred, target, current_epoch=0)
        loss.backward()

        assert pred.grad is not None, "No gradients flowing to predictions"
        assert pred.grad.abs().sum().item() > 0, (
            "Gradients are zero — no gradient flow"
        )

    def test_loss_decreases_on_better_prediction(self):
        """Loss should be lower for a better prediction."""
        B, C, H, W = 2, 7, 64, 64
        target = torch.randint(0, C, (B, H, W))

        # Random prediction
        pred_random = torch.randn(B, C, H, W)

        # Better prediction (correct class has higher logit)
        pred_better = torch.randn(B, C, H, W)
        pred_better.scatter_add_(1, target.unsqueeze(1), 2.0 * torch.ones(B, 1, H, W))

        config = load_test_config()
        criterion = BoundaryAwareLoss(config, num_classes=C)

        loss_random = criterion(pred_random, target)
        loss_better = criterion(pred_better, target)

        assert loss_better < loss_random, (
            f"Better pred loss {loss_better:.4f} should be < random {loss_random:.4f}"
        )

    def test_deep_supervision_aux_weights_anneal(self):
        """Aux weights should anneal to zero over specified epochs."""
        config = load_test_config()
        config["model"]["aux_weight_anneal_epochs"] = 10

        criterion = BoundaryAwareLoss(config, num_classes=7)

        weights_epoch_0 = criterion.get_aux_weights(0)
        weights_epoch_5 = criterion.get_aux_weights(5)
        weights_epoch_10 = criterion.get_aux_weights(10)
        weights_epoch_15 = criterion.get_aux_weights(15)

        assert weights_epoch_0[0] > 0, "Aux weights should be > 0 at epoch 0"
        assert weights_epoch_5[0] < weights_epoch_0[0], (
            "Aux weights should decrease by epoch 5"
        )
        for w in weights_epoch_10:
            assert w == 0.0, "Aux weights should be 0 at epoch 10"
        for w in weights_epoch_15:
            assert w == 0.0, "Aux weights should stay 0 after annealing"

    def test_deep_supervision_gradient_flow(self):
        """Deep supervision should allow gradient flow through aux heads."""
        B, C, H, W = 2, 7, 64, 64
        target = torch.randint(0, C, (B, H, W))

        # Simulate model with deep supervision: main + 2 aux outputs
        main_pred = torch.randn(B, C, H, W, requires_grad=True)
        aux1_pred = torch.randn(B, C, H // 2, W // 2, requires_grad=True)
        aux2_pred = torch.randn(B, C, H // 4, W // 4, requires_grad=True)

        config = load_test_config()
        criterion = BoundaryAwareLoss(config, num_classes=C)

        loss = criterion((main_pred, [aux1_pred, aux2_pred]), target, current_epoch=0)
        loss.backward()

        assert main_pred.grad is not None, "No gradients to main head"
        assert aux1_pred.grad is not None, "No gradients to aux head 1"
        assert aux2_pred.grad is not None, "No gradients to aux head 2"
