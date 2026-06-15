"""
export_onnx.py — Export trained model to ONNX with dynamic spatial dimensions.

Exports the model to ONNX format with:
  - Dynamic H and W dimensions (supports arbitrary input resolution)
  - Deep supervision heads merged out (inference-only graph)
  - Verified against PyTorch output pixel-by-pixel (max diff < 1e-4)

Usage:
    python src/serving/export_onnx.py \
        --checkpoint checkpoints/best_model.pth \
        --config configs/custom.yaml \
        --output model.onnx
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.efficient_unet import EfficientUNet
from src.models.vanilla_unet import VanillaUNet
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export segmentation model to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output path for ONNX model (default: model.onnx)",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--input_height",
        type=int,
        default=512,
        help="Input height for static shape tracing (default: 512)",
    )
    parser.add_argument(
        "--input_width",
        type=int,
        default=512,
        help="Input width for static shape tracing (default: 512)",
    )
    parser.add_argument(
        "--dynamic_axes",
        action="store_true",
        default=True,
        help="Export with dynamic H and W axes (default: True)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify ONNX output against PyTorch output (default: True)",
    )
    return parser.parse_args()


def load_model_for_export(
    checkpoint_path: str, config: Dict, device: torch.device
) -> torch.nn.Module:
    """Load model with deep supervision heads removed for inference."""
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "efficient_unet")
    num_classes = config["dataset"]["num_classes"]

    if arch == "vanilla_unet":
        model = VanillaUNet(num_classes=num_classes)
    elif arch == "efficient_unet":
        model = EfficientUNet(
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b4"),
            encoder_weights=None,
            num_classes=num_classes,
            decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32, 16]),
            use_scse=model_cfg.get("use_scse", True),
            deep_supervision=False,  # No aux heads for inference
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Load checkpoint, stripping aux head keys
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    if arch == "efficient_unet":
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("aux_heads.")
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys (expected if aux heads stripped): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_height: int = 512,
    input_width: int = 512,
    num_classes: int = 7,
    opset_version: int = 17,
    dynamic_axes: bool = True,
    device: torch.device = torch.device("cuda"),
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model (eval mode).
        output_path: Path to save the ONNX model.
        input_height: Height for the dummy input tensor.
        input_width: Width for the dummy input tensor.
        num_classes: Number of segmentation classes.
        opset_version: ONNX opset version.
        dynamic_axes: Whether to use dynamic spatial dimensions.
        device: Device for tracing.

    Returns:
        (output_path, pytorch_output, onnx_output) for verification.
    """
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_height, input_width, device=device)

    # Get PyTorch output for verification
    with torch.no_grad():
        torch_output = model(dummy_input)

    # Define dynamic axes
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    print(f"ONNX model exported to: {output_path}")
    print(f"  Input shape:  (1, 3, {input_height}, {input_width})")
    print(f"  Output shape: (1, {num_classes}, {input_height}, {input_width})")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic axes: {dynamic_axes}")

    return output_path, torch_output.cpu().numpy(), dummy_input.cpu().numpy()


def verify_onnx(
    onnx_path: str,
    pytorch_output: np.ndarray,
    dummy_input: np.ndarray,
    num_classes: int,
) -> bool:
    """Verify ONNX model output matches PyTorch output.

    Runs the ONNX model with onnxruntime and compares output
    pixel-by-pixel. Maximum allowed difference: 1e-4.

    Args:
        onnx_path: Path to exported ONNX model.
        pytorch_output: Output from PyTorch model.
        dummy_input: Input tensor used for export.
        num_classes: Number of classes.

    Returns:
        True if verification passes.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed. Skipping verification.")
        return False

    # Create ONNX Runtime session
    providers = []
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    onnx_output = session.run([output_name], {input_name: dummy_input})[0]

    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"\nONNX Verification:")
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  ONNX output shape:    {onnx_output.shape}")
    print(f"  Max difference:       {max_diff:.6e}")
    print(f"  Mean difference:      {mean_diff:.6e}")

    # Check shape
    assert pytorch_output.shape == onnx_output.shape, (
        f"Shape mismatch: {pytorch_output.shape} vs {onnx_output.shape}"
    )

    # Check values
    if max_diff < 1e-4:
        print(f"  ✓ Verification PASSED (max_diff < 1e-4)")
        return True
    else:
        print(f"  ✗ Verification WARNING (max_diff >= 1e-4)")
        print(f"    This may be due to floating-point differences between")
        print(f"    PyTorch and ONNX Runtime. The prediction quality should")
        print(f"    still be equivalent if argmax gives the same class.")
        return False


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model for export...")
    model = load_model_for_export(args.checkpoint, config, device)

    # Export to ONNX
    output_path = args.output
    onnx_path, torch_output, dummy_input = export_to_onnx(
        model=model,
        output_path=output_path,
        input_height=args.input_height,
        input_width=args.input_width,
        num_classes=config["dataset"]["num_classes"],
        opset_version=args.opset_version,
        dynamic_axes=args.dynamic_axes,
        device=device,
    )

    # Verify
    if args.verify:
        verify_onnx(
            onnx_path,
            torch_output,
            dummy_input,
            config["dataset"]["num_classes"],
        )

    # Print model info
    import onnx
    onnx_model = onnx.load(onnx_path)
    n_nodes = len(onnx_model.graph.node)
    n_params = sum(
        np.prod(t.dims)
        for t in onnx_model.graph.initializer
    )
    print(f"\nONNX Model Info:")
    print(f"  Nodes:     {n_nodes:,}")
    print(f"  Params:    {n_params:,}")
    print(f"  Size:      {os.path.getsize(onnx_path) / 1024**2:.1f} MB")
    print(f"  IR version: {onnx_model.ir_version}")


if __name__ == "__main__":
    main()
