"""
visualization.py — Helper functions for segmentation visualisation.

Provides:
  - colourise_mask: Convert class indices to RGB
  - overlay_mask: Blend image with transparent segmentation overlay
  - plot_confusion_matrix: Normalised confusion matrix heatmap
  - plot_qualitative_grid: Side-by-side image / GT / pred comparison
  - plot_iou_comparison: Bar chart comparing two models' per-class IoU
  - log_wandb_images: Log segmentation overlays to wandb
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.data.dataset import LoveDA


def colourise_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class index mask (HxW, values 0-6) to RGB (HxWx3)."""
    return LoveDA.decode_mask(mask)


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay a colourised mask on an image.

    Args:
        image: (H, W, 3) RGB image in [0, 1].
        mask: (H, W) integer class indices.
        alpha: Transparency of the overlay.

    Returns:
        (H, W, 3) overlay image.
    """
    mask_rgb = colourise_mask(mask).astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * image + alpha * mask_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    normalize: bool = True,
) -> None:
    """Plot and save a normalised confusion matrix heatmap.

    Args:
        cm: (num_classes, num_classes) confusion matrix.
        class_names: List of class names.
        output_path: Path to save the figure.
        normalize: Whether to normalise rows (true class distribution).
    """
    if normalize:
        cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalised)" if normalize else ""))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_qualitative_grid(
    images: List[np.ndarray],
    masks_gt: List[np.ndarray],
    masks_pred: List[np.ndarray],
    output_path: str,
    n_samples: int = 8,
    model_name: str = "",
) -> None:
    """Plot side-by-side comparison of images, GT masks, and predicted masks.

    Args:
        images: List of (H, W, 3) RGB images.
        masks_gt: List of (H, W) ground truth masks.
        masks_pred: List of (H, W) predicted masks.
        output_path: Path to save the figure.
        n_samples: Number of samples to display.
        model_name: Model name for the title.
    """
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img = images[i]
        gt = masks_gt[i]
        pred = masks_pred[i]

        # Clamp image
        img = np.clip(img, 0, 1)

        # RGB masks
        gt_rgb = colourise_mask(gt)
        pred_rgb = colourise_mask(pred)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title(f"Prediction {model_name}", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_iou_bar_chart(
    iou_dict: Dict[str, List[float]],
    class_names: List[str],
    output_path: str,
    title: str = "Per-Class IoU Comparison",
) -> None:
    """Plot grouped bar chart comparing per-class IoU across models.

    Args:
        iou_dict: {model_name: [per_class_iou_list]}.
        class_names: List of class names.
        output_path: Path to save the figure.
        title: Chart title.
    """
    n_models = len(iou_dict)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_classes)
    width = 0.8 / n_models

    for i, (model_name, ious) in enumerate(iou_dict.items()):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, ious, width, label=model_name)
        for bar, val in zip(bars, ious):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def log_wandb_images(
    wandb_logger,
    images: List[np.ndarray],
    masks_gt: List[np.ndarray],
    masks_pred: List[np.ndarray],
    class_names: List[str],
    prefix: str = "val",
    max_samples: int = 8,
) -> None:
    """Log segmentation overlays to wandb.

    Args:
        wandb_logger: wandb module (import wandb).
        images: List of (H, W, 3) RGB images.
        masks_gt: List of (H, W) ground truth masks.
        masks_pred: List of (H, W) predicted masks.
        class_names: List of class names.
        prefix: Prefix for wandb panel name.
        max_samples: Max samples to log.
    """
    import wandb

    n = min(max_samples, len(images))
    wandb_images = []

    for i in range(n):
        img = np.clip(images[i], 0, 1)
        gt_rgb = colourise_mask(masks_gt[i])
        pred_rgb = colourise_mask(masks_pred[i])
        overlay = overlay_mask(img, masks_pred[i], alpha=0.4)

        combined = np.concatenate([img, gt_rgb / 255.0, pred_rgb / 255.0, overlay], axis=1)
        wandb_images.append(
            wandb.Image(
                (combined * 255).astype(np.uint8),
                caption=f"{prefix}_{i}",
            )
        )

    wandb_logger.log({f"{prefix}/segmentation_overlays": wandb_images})
