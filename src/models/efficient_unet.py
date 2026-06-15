"""
efficient_unet.py — EfficientNet-B4 U-Net with SCSE and deep supervision.

Architecture overview:
  ┌─────────────────────────────────────────────────────┐
  │                EfficientNet-B4 Encoder               │
  │  (pretrained on ImageNet, features_only=True)        │
  │                                                      │
  │  Input (3×H×W)                                       │
  │    │                                                 │
  │  Stage 0:  stride 2  →  C0 = 48  (H/2 × W/2)       │
  │  Stage 1:  stride 4  →  C1 = 32  (H/4 × W/4)       │
  │  Stage 2:  stride 8  →  C2 = 56  (H/8 × W/8)       │
  │  Stage 3:  stride 16 →  C3 = 160 (H/16 × W/16)     │
  │  Stage 4:  stride 32 →  C4 = 448 (H/32 × W/32)     │
  └──────┬──────┬──────┬──────┬──────┘                    │
         │      │      │      │                          │
         v      v      v      v                          │
  ┌──────────────────────────────────────────────────────┤
  │  Decoder (top-down with skip connections)             │
  │                                                      │
  │          ┌───────────────────────────┐               │
  │          │ DecoderBlock 0 (C4 → D0)  │ ← Skip C3     │
  │          │   + SCSEBlock            │               │
  │          ├───────────────────────────┤               │
  │          │ DecoderBlock 1 (D0 → D1)  │ ← Skip C2     │
  │          │   + SCSEBlock            │               │
  │          ├───────────────────────────┤               │
  │          │ DecoderBlock 2 (D1 → D2)  │ ← Skip C1     │
  │          │   + SCSEBlock            │               │
  │          ├───────────────────────────┤               │
  │          │ DecoderBlock 3 (D2 → D3)  │ ← Skip C0     │
  │          │   + SCSEBlock            │               │
  │          └──────────┬────────────────┘               │
  │                     │                                │
  │           SegmentationHead (D3 → num_classes)        │
  │                     │                                │
  │               Prediction (num_classes × H × W)       │
  └──────────────────────────────────────────────────────┘

Deep supervision:
  Auxiliary segmentation heads attached after decoder blocks 0 and 1,
  each producing a downsampled prediction. During training, the loss is:
      L = w0 * L_main + w1 * L_aux1 + w2 * L_aux2
  Auxiliary weights are annealed to zero over the first N epochs.
  At inference, only the main head is used.

  Why deep supervision:
    - Provides gradient signal to intermediate decoder layers
    - Improves convergence, especially with frozen encoder in early epochs
    - Forces intermediate features to be semantically meaningful
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm

from src.models.blocks import (
    ConvBlock,
    DecoderBlock,
    SCSEBlock,
    SegmentationHead,
)


class EfficientUNet(nn.Module):
    """
    U-Net with EfficientNet-B4 encoder, SCSE blocks, and deep supervision.

    Args:
        encoder_name: timm model name (default: efficientnet-b4).
        encoder_weights: Pretrained weights (default: imagenet).
        num_classes: Number of segmentation classes (default: 7).
        decoder_channels: Channel dimensions for decoder levels (top-down).
                          Default: [256, 128, 64, 32, 16]
        use_scse: Whether to add SCSE blocks after each decoder block.
        deep_supervision: Whether to add auxiliary segmentation heads.
        deep_supervision_weights: Loss weights for [main, aux1, aux2].
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        num_classes: int = 7,
        decoder_channels: Optional[List[int]] = None,
        use_scse: bool = True,
        deep_supervision: bool = True,
        deep_supervision_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.use_scse = use_scse
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights or [1.0, 0.5, 0.25]
        decoder_channels = decoder_channels or [256, 128, 64, 32, 16]

        # ── Encoder ────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=(encoder_weights == "imagenet"),
            features_only=True,
        )

        # Extract encoder channel dimensions dynamically
        # feature_info is a list of dicts with 'num_chs', 'reduction', etc.
        encoder_channels = self.encoder.feature_info.channels()
        # Typically for EfficientNet-B4: [48, 32, 56, 160, 448]
        self.encoder_channels = encoder_channels

        # Number of feature levels used (skip connections)
        self.num_encoder_levels = len(encoder_channels)

        # ── Decoder ────────────────────────────────────────────────────
        # Build top-down: from lowest resolution to highest
        dec_channels = decoder_channels[:self.num_encoder_levels]
        self.decoder_blocks = nn.ModuleList()

        for i in range(self.num_encoder_levels - 1, -1, -1):
            if i == self.num_encoder_levels - 1:
                # Bottom level: no upsample, just process encoder features
                in_ch = encoder_channels[i]
                skip_ch = 0
            else:
                # Concatenate: upsampled decoder output + encoder skip
                in_ch = dec_channels[i + 1]
                skip_ch = encoder_channels[i]

            out_ch = dec_channels[i]

            block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
            )
            self.decoder_blocks.append(block)

            if self.use_scse:
                self.add_module(f"scse_{i}", SCSEBlock(out_ch))

        # ── Segmentation heads ─────────────────────────────────────────
        self.seg_head = SegmentationHead(dec_channels[0], num_classes)

        if self.deep_supervision:
            self.aux_heads = nn.ModuleList()
            for i in range(min(self.num_encoder_levels - 1, 3)):
                aux_ch = min(dec_channels[i], dec_channels[0])
                self.aux_heads.append(SegmentationHead(aux_ch, num_classes))

        # ── Encoder projection for bottleneck (if channels differ) ─────
        if encoder_channels[-1] != dec_channels[-1]:
            self.bottleneck_proj = nn.Conv2d(
                encoder_channels[-1], dec_channels[-1], kernel_size=1, bias=False
            )
        else:
            self.bottleneck_proj = nn.Identity()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            If deep_supervision and training:
                (main_pred, [aux1_pred, aux2_pred])
            Else:
                main_pred (B, num_classes, H, W)
        """
        # Encoder: extract multi-scale features
        encoder_features = self.encoder(x)
        # encoder_features[0]: high-res (stride 2)
        # encoder_features[-1]: low-res (stride 32)

        # Decoder: top-down with skip connections
        decoder_outputs: List[torch.Tensor] = []

        # Process from bottom to top
        for i in range(self.num_encoder_levels - 1, -1, -1):
            decoder_idx = self.num_encoder_levels - 1 - i

            if i == self.num_encoder_levels - 1:
                # Bottleneck: from encoder bottom
                dec_out = self.bottleneck_proj(encoder_features[i])
            else:
                skip = encoder_features[i]
                dec_out = self.decoder_blocks[decoder_idx](
                    decoder_outputs[-1], skip
                )

            if self.use_scse:
                scse = getattr(self, f"scse_{i}")
                dec_out = scse(dec_out)

            decoder_outputs.append(dec_out)

        # Main segmentation head (highest resolution)
        main_out = self.seg_head(decoder_outputs[-1])

        if self.deep_supervision:
            aux_outputs = []
            for idx, aux_head in enumerate(self.aux_heads):
                aux_idx = min(idx + 1, len(decoder_outputs) - 1)
                aux_out = aux_head(decoder_outputs[aux_idx])
                # Upsample to match main output resolution
                if aux_out.shape[2:] != main_out.shape[2:]:
                    aux_out = nn.functional.interpolate(
                        aux_out,
                        size=main_out.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                aux_outputs.append(aux_out)
            return main_out, aux_outputs

        return main_out

    def get_encoder_params(self) -> List[torch.nn.Parameter]:
        """Return encoder parameters (for differential learning rates)."""
        return list(self.encoder.parameters())

    def get_decoder_params(self) -> List[torch.nn.Parameter]:
        """Return decoder and head parameters."""
        decoder_params = list(self.decoder_blocks.parameters())
        if hasattr(self, "bottleneck_proj"):
            decoder_params.extend(self.bottleneck_proj.parameters())
        decoder_params.extend(self.seg_head.parameters())
        if self.deep_supervision:
            decoder_params.extend(self.aux_heads.parameters())
        if self.use_scse:
            for i in range(self.num_encoder_levels):
                scse = getattr(self, f"scse_{i}", None)
                if scse is not None:
                    decoder_params.extend(scse.parameters())
        return decoder_params

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
