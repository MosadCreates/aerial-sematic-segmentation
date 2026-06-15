"""
benchmark.py — Benchmark inference latency across PyTorch and ONNX Runtime.

Measures:
  - PyTorch FP32 latency
  - PyTorch FP16 latency (when CUDA is available)
  - ONNX Runtime latency (CPU or CUDA)

Each measurement is over 100 runs, reporting mean ± std latency
for both a single 1024×1024 image and a batch of 4.

Results are saved to results/inference_benchmark.json.

Usage:
    python src/serving/benchmark.py \
        --checkpoint checkpoints/best_model.pth \
        --config configs/custom.yaml \
        --output results/inference_benchmark.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.efficient_unet import EfficientUNet
from src.models.vanilla_unet import VanillaUNet
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency"
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
        default="results/inference_benchmark.json",
        help="Output path for benchmark results (default: results/inference_benchmark.json)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of runs for each configuration (default: 100)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=20,
        help="Number of warmup runs before measurement (default: 20)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Batch sizes to benchmark (default: 1 4)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size (H=W) for benchmarking (default: 1024)",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: str, config: Dict, device: torch.device
) -> torch.nn.Module:
    """Load model for inference benchmarking."""
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
            deep_supervision=False,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if hasattr(model, "aux_heads"):
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("aux_heads.")}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def measure_latency(
    model_fn,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 20,
    amp: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Dict:
    """Measure inference latency.

    Args:
        model_fn: Callable that takes input and returns output.
        input_tensor: Input tensor for inference.
        num_runs: Number of measurement runs.
        warmup_runs: Number of warmup runs.
        amp: Whether to use automatic mixed precision (FP16).
        device: Device.

    Returns:
        Dict with mean, std, min, max, p50, p95, p99 latency in ms.
    """
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            with torch.cuda.amp.autocast(enabled=amp):
                _ = model_fn(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
                start = time.perf_counter()
            else:
                start = time.perf_counter()

            with torch.cuda.amp.autocast(enabled=amp):
                _ = model_fn(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def benchmark_onnx(
    onnx_path: str,
    input_tensor: np.ndarray,
    num_runs: int = 100,
    warmup_runs: int = 20,
) -> Dict:
    """Benchmark ONNX Runtime inference latency."""
    import onnxruntime as ort

    providers = []
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Warmup
    for _ in range(warmup_runs):
        _ = session.run([output_name], {input_name: input_tensor})

    # Measurement
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run([output_name], {input_name: input_tensor})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, config, device)
    arch = config["model"].get("architecture", "unknown")

    results = {
        "model": arch,
        "device": str(device),
        "image_size": args.image_size,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "configs": {},
    }

    for batch_size in args.batch_sizes:
        key = f"batch_{batch_size}"
        results["configs"][key] = {}
        input_tensor = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)

        # PyTorch FP32
        print(f"\nBenchmarking batch_size={batch_size}...")
        print(f"  PyTorch FP32...", end=" ", flush=True)
        model_fn = lambda x: model(x)
        fp32_results = measure_latency(
            model_fn, input_tensor, args.num_runs, args.warmup_runs, amp=False, device=device
        )
        results["configs"][key]["pytorch_fp32"] = fp32_results
        print(f"{fp32_results['mean_ms']:.2f} ± {fp32_results['std_ms']:.2f} ms")

        # PyTorch FP16 (CUDA only)
        if device.type == "cuda":
            print(f"  PyTorch FP16...", end=" ", flush=True)
            fp16_results = measure_latency(
                model_fn, input_tensor, args.num_runs, args.warmup_runs, amp=True, device=device
            )
            results["configs"][key]["pytorch_fp16"] = fp16_results
            print(f"{fp16_results['mean_ms']:.2f} ± {fp16_results['std_ms']:.2f} ms")

        # ONNX Runtime
        onnx_path = "model.onnx"
        if os.path.exists(onnx_path):
            try:
                print(f"  ONNX Runtime...", end=" ", flush=True)
                onnx_results = benchmark_onnx(
                    onnx_path, input_tensor.cpu().numpy(), args.num_runs, args.warmup_runs
                )
                results["configs"][key]["onnx"] = onnx_results
                print(f"{onnx_results['mean_ms']:.2f} ± {onnx_results['std_ms']:.2f} ms")
            except Exception as e:
                print(f"ONNX benchmark failed: {e}")

    # Save results
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to: {output_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Summary (mean ± std ms):")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'BS=1':<20s} {'BS=4':<20s}")
    print(f"{'-'*25} {'-'*20} {'-'*20}")

    for config_name in ["pytorch_fp32", "pytorch_fp16", "onnx"]:
        row = f"{config_name:<25s}"
        for bs in args.batch_sizes:
            key = f"batch_{bs}"
            if config_name in results["configs"].get(key, {}):
                r = results["configs"][key][config_name]
                row += f" {r['mean_ms']:.1f} ± {r['std_ms']:.1f} ms  "
            else:
                row += f" {'N/A':<20s}"
        print(row)


if __name__ == "__main__":
    main()
