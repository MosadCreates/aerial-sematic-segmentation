"""
blocks.py — Reusable building blocks for the U-Net family.

Components:
  - ConvBlock: Conv2d + BN + ReLU (double conv for U-Net)
  - DecoderBlock: bilinear upsample → concatenate skip → ConvBlock
  - SCSEBlock: Spatial + Channel Squeeze-and-Excitation
  - SegmentationHead: 1x1 conv to num_classes

DecoderBlock uses bilinear upsampling instead of transposed convolution:
  - Bilinear upsampling has no learnable parameters → fewer params, less overfitting
  - Transposed conv can produce checkerboard artefacts; bilinear avoids this
  - The subsequent Conv-BN-ReLU layers are sufficient to learn the upsampling function

SCSEBlock (Spatial and Channel Squeeze-and-Excitation):
  Proposed in "Concurrent Spatial and Channel Squeeze & Excitation in Fully
  Convolutional Networks" (Roy et al., 2018).
  - cSE: Global context vector → FC layers → recalibrates channels
  - sSE: 1x1 conv → spatial attention map → recalibrates spatial locations
  - SCSE = element-wise sum of cSE and sSE recalibrated features
  Why this improves segmentation: aerial classes have both channel-specific
  (e.g., water is spectrally distinct) and location-specific patterns
  (e.g., roads have linear spatial structure). SCSE addresses both.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two sequential Conv2d-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net decoder block:
      1. Bilinear upsample 2x
      2. Concatenate with encoder skip connection
      3. Double Conv-BN-ReLU

    Uses bilinear upsampling (not transposed conv) because:
      - No learnable parameters (lighter, less overfitting)
      - No checkerboard artefacts
      - The double conv after concatenation learns the upsampling function
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = ConvBlock(
            in_channels + skip_channels, out_channels
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SCSEBlock(nn.Module):
    """
    Spatial and Channel Squeeze-and-Excitation block.

    cSE (Channel recalibration):
      GlobalAvgPool → FC(C/r) → ReLU → FC(C) → Sigmoid → scale channels

    sSE (Spatial recalibration):
      Conv2d(C→1, 1×1) → Sigmoid → scale spatial locations

    SCSE = input * cSE_weight + input * sSE_weight
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel SE
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # Spatial SE
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cse(x) + x * self.sse(x)


class SegmentationHead(nn.Module):
    """
    Segmentation head: 1x1 conv to num_classes.

    No activation — outputs raw logits for loss computation.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
