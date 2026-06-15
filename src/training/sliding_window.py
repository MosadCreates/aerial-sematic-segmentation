"""
sliding_window.py — Sliding window inference for full-resolution images.

LoveDA images are 1024×1024, but models are trained on 512×512 crops.
During validation/testing, we can't simply resize to 512 — that would lose
the fine detail needed for accurate evaluation. Instead, we use a sliding
window approach:

  1. Divide the full-resolution image into overlapping patches (512×512)
  2. Run inference on each patch
  3. Reconstruct the full prediction by:
     a. Averaging overlapping regions (uniform weighting)
     b. Gaussian weighting: patches contribute more at their centre,
        less at edges — this reduces boundary artefacts between patches

Why not just upscale the 512×512 prediction to 1024×1024?
  - Upscaling loses high-frequency boundary detail (classes are per-pixel)
  - Sliding window preserves the full-resolution prediction quality

Gaussian weighting at patch boundaries:
  A Gaussian weight map is applied to each patch's prediction before
  accumulation. The weights are highest at the patch centre and smoothly
  decay to zero at the edges. This prevents hard seams between patches
  and produces cleaner full-resolution predictions.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def gaussian_weight_map(
    height: int, width: int, sigma: float = 0.25
) -> torch.Tensor:
    """Create a 2D Gaussian weight map for blending overlapping patches.

    The Gaussian is centred in the patch and decays to near-zero at edges.
    Sigma is relative to patch size (fraction of min dimension).

    Args:
        height: Patch height.
        width: Patch width.
        sigma: Gaussian sigma as fraction of min(H, W) (default: 0.25).

    Returns:
        (height, width) tensor with values in [0, 1].
    """
    min_dim = min(height, width)
    sigma_px = sigma * min_dim
    center_y, center_x = height / 2.0, width / 2.0

    y = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(width, dtype=torch.float32).unsqueeze(0)

    weight = torch.exp(
        -((y - center_y) ** 2 + (x - center_x) ** 2) / (2.0 * sigma_px ** 2)
    )
    return weight


def sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: int = 512,
    stride: int = 256,
    num_classes: int = 7,
    gaussian_weight: bool = True,
    device: torch.device = torch.device("cuda"),
    mixed_precision: bool = True,
) -> np.ndarray:
    """Run sliding window inference on a single full-resolution image.

    Args:
        model: Segmentation model.
        image: (3, H, W) tensor, normalised with ImageNet stats.
        patch_size: Size of inference patches (default: 512).
        stride: Stride between patches (default: 256 = 50% overlap).
        num_classes: Number of segmentation classes.
        gaussian_weight: Whether to use Gaussian blending (default: True).
        device: Device for inference.
        mixed_precision: Whether to use FP16 autocast.

    Returns:
        (H, W) numpy array of class index predictions.
    """
    model.eval()
    _, H, W = image.shape
    image = image.unsqueeze(0).to(device)  # (1, 3, H, W)

    # Accumulator for votes and weight normalisation
    prediction_map = torch.zeros((num_classes, H, W), device=device)
    weight_map = torch.zeros((1, H, W), device=device)

    if gaussian_weight:
        patch_weight = gaussian_weight_map(patch_size, patch_size)
        patch_weight = patch_weight.to(device)
    else:
        patch_weight = torch.ones((patch_size, patch_size), device=device)

    # Compute patches
    y_positions = list(range(0, H - patch_size + 1, stride))
    x_positions = list(range(0, W - patch_size + 1, stride))

    # Ensure coverage: add final patch if needed
    if (H - patch_size) % stride != 0:
        y_positions.append(H - patch_size)
    if (W - patch_size) % stride != 0:
        x_positions.append(W - patch_size)

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                patch = image[:, :, y:y + patch_size, x:x + patch_size]

                with torch.cuda.amp.autocast(enabled=mixed_precision):
                    output = model(patch)

                if isinstance(output, (list, tuple)):
                    output = output[0]  # Main head (ignore aux for inference)

                # Softmax to probabilities
                probs = torch.softmax(output, dim=1).squeeze(0)  # (C, patch, patch)

                # Apply patch weight and accumulate
                prediction_map[:, y:y + patch_size, x:x + patch_size] += (
                    probs * patch_weight
                )
                weight_map[:, y:y + patch_size, x:x + patch_size] += patch_weight

    # Normalise by accumulated weights and convert to class indices
    prediction_map = prediction_map / (weight_map + 1e-8)
    pred_class = prediction_map.argmax(dim=0).cpu().numpy().astype(np.int64)

    return pred_class
