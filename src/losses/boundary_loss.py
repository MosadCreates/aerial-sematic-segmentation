"""
boundary_loss.py — Boundary-aware loss component for segmentation.

Motivation:
  Boundary regions are the hardest to segment in aerial imagery:
  - Class boundaries occupy few pixels but contribute most to prediction errors
  - Buildings have sharp rectilinear boundaries; roads have thin linear structure
  - Water and vegetation boundaries are irregular and ambiguous
  - Standard loss functions (CE, Dice) treat all pixels equally, so boundary
    errors get diluted by the majority of interior pixels

Approach:
  1. Compute ground-truth boundary maps using morphological operations:
       boundary = dilation(mask) - erosion(mask)
     This extracts pixels that are within kernel_size/2 of a class transition.
  2. Assign a higher loss weight to boundary pixels (configurable multiplier).
  3. The total boundary loss is the mean loss weighted by boundary importance.

Implementation details:
  - Morphological operations use a square kernel of size kernel_size.
  - boundary_width (K) controls how many pixels from a boundary are up-weighted.
  - boundary_weight controls the magnitude of the penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def morphological_boundary(
    mask: torch.Tensor,
    kernel_size: int = 3,
    num_classes: int = 7,
) -> torch.Tensor:
    """
    Compute boundary maps from segmentation masks using morphological erosion/dilation.

    For each class, the boundary is defined as the set difference between
    the dilated class region and the eroded class region:
        boundary_c = dilation(mask_c) - erosion(mask_c)

    The final boundary map is the union of boundaries across all classes.

    Args:
        mask: (B, H, W) integer class indices.
        kernel_size: Size of the morphological kernel.
        num_classes: Number of segmentation classes.

    Returns:
        boundary_map: (B, H, W) float tensor with 1 at boundary pixels, 0 elsewhere.
    """
    B, H, W = mask.shape
    device = mask.device

    # One-hot encode mask: (B, C, H, W)
    mask_one_hot = torch.zeros(B, num_classes, H, W, device=device)
    mask_one_hot.scatter_(1, mask.unsqueeze(1), 1.0)

    # Create morphological kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    kernel = kernel / kernel.sum()  # Normalise for mean filter

    # Pad to preserve spatial size
    pad = kernel_size // 2

    # Dilate and erode via max-pooling and min-pooling
    # For binary masks, dilation = max pool, erosion = min pool
    dilated = F.max_pool2d(mask_one_hot, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask_one_hot, kernel_size, stride=1, padding=pad)

    # Boundary = dilated - eroded (for each class)
    boundary = dilated - eroded  # (B, C, H, W)

    # Union across classes: pixel is boundary if it's a boundary for any class
    boundary_map = boundary.sum(dim=1).clamp(0, 1)

    return boundary_map


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that up-weights pixels near class boundaries.

    Args:
        kernel_size: Size of morphological kernel (default: 3).
        boundary_width: Number of pixels from boundary to up-weight (default: 3).
        boundary_weight: Multiplier for boundary pixel loss (default: 2.0).
        num_classes: Number of segmentation classes (default: 7).
    """

    def __init__(
        self,
        kernel_size: int = 3,
        boundary_width: int = 3,
        boundary_weight: float = 2.0,
        num_classes: int = 7,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.num_classes = num_classes

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weight base_loss by boundary importance.

        Args:
            pred: (B, C, H, W) raw logits (not used directly, but kept for API consistency).
            target: (B, H, W) integer class indices.
            base_loss: (B, H, W) per-pixel loss from CE or Dice.

        Returns:
            Scalar boundary-weighted loss.
        """
        with torch.no_grad():
            boundary_map = morphological_boundary(
                target, self.kernel_size, self.num_classes
            )

            # Create boundary weight map
            # Pixels within boundary_width of a class boundary get multiplied
            weight_map = torch.ones_like(boundary_map)
            weight_map[boundary_map > 0] = self.boundary_weight

            # Normalise: average weight = 1 to keep loss magnitude stable
            weight_map = weight_map / weight_map.mean()

        # Apply boundary weights to per-pixel loss
        weighted_loss = base_loss * weight_map
        return weighted_loss.mean()
