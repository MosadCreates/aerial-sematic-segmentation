"""
model_summary.py — Print model architecture summary.

Computes and prints:
  - Total parameters
  - Trainable parameters
  - Encoder vs decoder parameter counts
  - Forward-pass shape trace through all layers

Usage:
    python src/models/model_summary.py --config configs/custom.yaml
    python src/models/model_summary.py --config configs/baseline.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import yaml

from src.models.efficient_unet import EfficientUNet
from src.models.vanilla_unet import VanillaUNet
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print model architecture summary"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Input spatial size (H W) for forward pass trace",
    )
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def forward_shape_trace(model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
    """Print the shape of each layer's output in the forward pass.

    Uses forward hooks to trace tensor shapes through the model.
    """
    shapes = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, (list, tuple)):
                out_shape = [o.shape for o in out if isinstance(o, torch.Tensor)]
            else:
                out_shape = out.shape
            shapes[name] = out_shape
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    print(f"\n  {'Layer':<45s} {'Output Shape':<20s}")
    print(f"  {'─'*45} {'─'*20}")
    for name, shape in shapes.items():
        print(f"  {name:<45s} {str(shape):<20s}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "efficient_unet")
    num_classes = model_cfg.get("num_classes", 7)

    print(f"Architecture: {arch}")
    print(f"Num classes:  {num_classes}")
    print("=" * 60)

    if arch == "efficient_unet":
        model = EfficientUNet(
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b4"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            num_classes=num_classes,
            decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32, 16]),
            use_scse=model_cfg.get("use_scse", True),
            deep_supervision=model_cfg.get("deep_supervision", True),
            deep_supervision_weights=model_cfg.get(
                "deep_supervision_weights", [1.0, 0.5, 0.25]
            ),
        )
    elif arch == "vanilla_unet":
        model = VanillaUNet(
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Count parameters
    total_params, trainable_params = count_params(model)

    if arch == "efficient_unet":
        encoder_params = sum(p.numel() for p in model.get_encoder_params())
        decoder_params = sum(p.numel() for p in model.get_decoder_params())
        frozen_params = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        print(f"\n  Parameter breakdown:")
        print(f"    Total params:      {total_params:>12,}")
        print(f"    Trainable params:  {trainable_params:>12,}")
        print(f"    Frozen params:     {frozen_params:>12,}")
        print(f"    Encoder params:    {encoder_params:>12,}")
        print(f"    Decoder params:    {decoder_params:>12,}")
        print(f"    Encoder %:         {encoder_params/total_params*100:>11.1f}%")
        print(f"    Decoder %:         {decoder_params/total_params*100:>11.1f}%")
    else:
        print(f"\n  Parameter breakdown:")
        print(f"    Total params:      {total_params:>12,}")
        print(f"    Trainable params:  {trainable_params:>12,}")

    # Encoder feature channels (dynamic from timm)
    if arch == "efficient_unet":
        print(f"\n  Encoder feature pyramid:")
        print(f"    {'Level':<10s} {'Stride':<10s} {'Channels':<10s}")
        print(f"    {'─'*10} {'─'*10} {'─'*10}")
        for i, ch in enumerate(model.encoder_channels):
            stride = 2 ** (i + 1)
            print(f"    {i:<10d} {stride:<10d} {ch:<10d}")

    # Forward shape trace
    H, W = args.input_size
    x = torch.randn(1, 3, H, W)
    print(f"\n  Forward shape trace (input: 1×3×{H}×{W}):")
    print(f"  {'─'*70}")

    with torch.no_grad():
        output = model(x)

    if isinstance(output, (list, tuple)):
        print(f"  Main output shape:   {output[0].shape}")
        for i, aux in enumerate(output[1]):
            print(f"  Aux {i+1} output shape:  {aux.shape}")
    else:
        print(f"  Output shape:        {output.shape}")

    if arch == "efficient_unet":
        print(f"\n  Deep supervision: {model.deep_supervision}")
        print(f"  SCSE blocks:      {model.use_scse}")


if __name__ == "__main__":
    main()
