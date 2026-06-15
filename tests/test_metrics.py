"""
test_metrics.py — Unit tests for metrics computation.

Tests:
  1. ConfusionMatrix: TP/FP/FN counts are correct
  2. mIoU: perfect prediction gives 1.0
  3. mIoU: completely wrong prediction gives 0.0
  4. Per-class IoU matches known values
  5. Pixel accuracy calculation
  6. F1 (Dice) relationship with IoU
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.evaluation.metrics import (
    ConfusionMatrix,
    compute_confusion_matrix,
    compute_all_metrics,
)


class TestConfusionMatrix:
    def test_perfect_prediction(self):
        """Perfect prediction should give IoU=1.0 and accuracy=1.0."""
        cm = ConfusionMatrix(num_classes=7)
        pred = np.array([[0, 1], [2, 3]])
        target = np.array([[0, 1], [2, 3]])
        cm.update(pred, target)

        iou = cm.compute_iou()
        assert np.allclose(iou[:4], 1.0), f"Perfect pred IoU should be 1.0, got {iou[:4]}"
        assert cm.compute_pixel_accuracy() == 1.0

    def test_completely_wrong_prediction(self):
        """Completely wrong prediction should give IoU=0.0 and accuracy=0.0."""
        cm = ConfusionMatrix(num_classes=3)
        pred = np.array([[1, 1], [1, 1]])
        target = np.array([[0, 0], [0, 0]])
        cm.update(pred, target)

        iou = cm.compute_iou()
        # IoU for class 0 should be 0 (no TP)
        assert iou[0] == 0.0, f"Wrong pred IoU for class 0 should be 0, got {iou[0]}"
        # IoU for class 1 should be 0 (no TP, since target has no class 1)
        # Wait, actually pred says class 1 everywhere, target says class 0 everywhere
        # TP for class 1 = 0, FP = 4, FN = 0
        assert iou[1] == 0.0, f"Wrong pred IoU for class 1 should be 0, got {iou[1]}"
        assert cm.compute_pixel_accuracy() == 0.0

    def test_multiple_updates(self):
        """Multiple updates should accumulate correctly."""
        cm = ConfusionMatrix(num_classes=2)

        pred1 = np.array([[0, 0], [0, 0]])
        target1 = np.array([[0, 0], [0, 0]])
        cm.update(pred1, target1)
        # TP0=4, FP0=0, FN0=0
        # TP1=0, FP1=0, FN1=0 (but denominator = 1 to avoid div by zero)

        pred2 = np.array([[1, 1], [1, 1]])
        target2 = np.array([[1, 1], [1, 1]])
        cm.update(pred2, target2)
        # TP0=4, FP0=0, FN0=0
        # TP1=4, FP1=0, FN1=0

        iou = cm.compute_iou()
        assert np.allclose(iou, 1.0), f"After two perfect updates, IoU should be 1.0"

    def test_miou_exclude_background(self):
        """Excluding background should change mIoU."""
        cm = ConfusionMatrix(num_classes=3)
        pred = np.array([[0, 0], [1, 2]])
        target = np.array([[0, 0], [1, 2]])
        cm.update(pred, target)

        miou_all = cm.compute_miou(exclude_background=False)
        miou_no_bg = cm.compute_miou(exclude_background=True)
        assert miou_all == 1.0, f"All perfect should give 1.0 mIoU"
        assert miou_no_bg == 1.0, f"Excluding bg on perfect should give 1.0"

        # Now test with incorrect background prediction
        cm2 = ConfusionMatrix(num_classes=3)
        pred2 = np.array([[1, 1], [1, 2]])
        target2 = np.array([[0, 0], [1, 2]])
        cm2.update(pred2, target2)

        miou_all2 = cm2.compute_miou(exclude_background=False)
        miou_no_bg2 = cm2.compute_miou(exclude_background=True)
        # Excluding bg should give higher mIoU since bg is wrong
        assert miou_no_bg2 >= miou_all2, (
            f"Excluding bg should increase mIoU when bg is inaccurate"
        )

    def test_pixel_accuracy_partial(self):
        """Pixel accuracy should reflect correct ratio."""
        cm = ConfusionMatrix(num_classes=2)
        pred = np.array([[0, 0], [1, 1]])
        target = np.array([[0, 1], [0, 1]])
        # TP0=1, TP1=1, total_correct=2, total_pixels=4
        cm.update(pred, target)
        assert cm.compute_pixel_accuracy() == 0.5, (
            f"Expected 0.5, got {cm.compute_pixel_accuracy()}"
        )

    def test_mean_accuracy(self):
        """Mean accuracy should average per-class recall."""
        cm = ConfusionMatrix(num_classes=2)
        pred = np.array([[0, 0], [0, 0]])
        target = np.array([[0, 0], [1, 1]])
        # Class 0: TP=2, FN=0 → acc=1.0
        # Class 1: TP=0, FN=2 → acc=0.0
        cm.update(pred, target)
        mean_acc = cm.compute_mean_accuracy()
        assert mean_acc == 0.5, f"Expected 0.5, got {mean_acc}"

    def test_f1_relationship_with_iou(self):
        """F1 (Dice) should be >= IoU for the same prediction."""
        cm = ConfusionMatrix(num_classes=3)
        rng = np.random.RandomState(42)
        pred = rng.randint(0, 3, (100, 100))
        target = rng.randint(0, 3, (100, 100))
        cm.update(pred, target)

        iou = cm.compute_iou()
        f1 = cm.compute_f1()
        for c in range(3):
            assert f1[c] >= iou[c], (
                f"F1[{c}] should be >= IoU[{c}]: F1={f1[c]:.4f}, IoU={iou[c]:.4f}"
            )


class TestComputeConfusionMatrix:
    def test_confusion_matrix_values(self):
        """Confusion matrix should count correct pairings."""
        pred = np.array([0, 1, 2, 0])
        target = np.array([0, 0, 2, 1])
        cm = compute_confusion_matrix(pred, target, num_classes=3)
        # cm[i, j] = count of true i predicted as j
        # (0,0): 1, (0,1): 1, (1,0): 1, (2,2): 1
        assert cm[0, 0] == 1, f"Expected 1, got {cm[0, 0]}"
        assert cm[0, 1] == 1, f"Expected 1, got {cm[0, 1]}"
        assert cm[1, 0] == 1, f"Expected 1, got {cm[1, 0]}"
        assert cm[2, 2] == 1, f"Expected 1, got {cm[2, 2]}"
        assert cm.sum() == 4, f"Total should be 4, got {cm.sum()}"


class TestComputeAllMetrics:
    def test_all_metrics_return_keys(self):
        """compute_all_metrics should return expected keys."""
        cm = ConfusionMatrix(num_classes=3)
        rng = np.random.RandomState(42)
        pred = rng.randint(0, 3, (50, 50))
        target = rng.randint(0, 3, (50, 50))
        cm.update(pred, target)

        class_names = ["bg", "road", "building"]
        metrics = compute_all_metrics(cm, class_names)

        assert "miou" in metrics
        assert "pixel_accuracy" in metrics
        assert "mean_accuracy" in metrics
        for name in class_names:
            assert f"{name}/iou" in metrics
            assert f"{name}/f1" in metrics
