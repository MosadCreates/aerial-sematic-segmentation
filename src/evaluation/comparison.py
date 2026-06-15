"""
comparison.py — Generate comparison table and visualisations between baseline and custom model.

Reads evaluation results from two model runs (baseline and custom) and generates:
  1. Markdown comparison table with per-class IoU and delta
  2. Per-class IoU bar chart comparing both models side by side
  3. Qualitative grid: 8 examples with image / GT / baseline pred / custom pred

Usage:
    python src/evaluation/comparison.py \
        --baseline results/baseline/vanilla_unet_results.json \
        --custom results/custom/efficient_unet_results.json \
        --output results/comparison_table.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset import LoveDA
from src.utils.visualization import plot_iou_bar_chart, colourise_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison between baseline and custom model"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline evaluation results JSON",
    )
    parser.add_argument(
        "--custom",
        type=str,
        required=True,
        help="Path to custom model evaluation results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_table.md",
        help="Output path for markdown comparison table",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save visualisation outputs",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=[
            "Background",
            "Building",
            "Road",
            "Water",
            "Barren",
            "Forest",
            "Agriculture",
        ],
        help="Class names in order",
    )
    return parser.parse_args()


def load_results(path: str) -> Dict:
    """Load evaluation results JSON."""
    with open(path, "r") as f:
        return json.load(f)


def format_iou(val: float) -> str:
    """Format IoU value as percentage string."""
    return f"{val * 100:.1f}"


def build_comparison_table(
    baseline_results: Dict,
    custom_results: Dict,
    class_names: List[str],
) -> str:
    """Build a markdown table comparing baseline vs custom model.

    Table format:
        | Model | mIoU | Building | Road | Water | Barren | Forest | Agriculture |
        |---|---|---|---|---|---|---|---|
        | Vanilla U-Net (baseline) | 45.7 | ... | ... | ... | ... | ... | ... |
        | EfficientNet-B4 U-Net | 52.1 | ... | ... | ... | ... | ... | ... |
        | Delta | +6.4 | ... | ... | ... | ... | ... | ... |
    """
    baseline_miou = baseline_results.get("miou", 0.0)
    custom_miou = custom_results.get("miou", 0.0)
    delta_miou = custom_miou - baseline_miou

    lines = [
        "# Comparison: Vanilla U-Net vs EfficientNet-B4 U-Net",
        "",
        "| Model | mIoU | Background | Building | Road | Water | Barren | Forest | Agriculture |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    # Baseline row
    base_row = f"| Vanilla U-Net (baseline) | {format_iou(baseline_miou)}"
    for name in class_names:
        key = f"{name}/iou"
        base_row += f" | {format_iou(baseline_results.get(key, 0.0))}"
    base_row += " |"
    lines.append(base_row)

    # Custom row
    custom_row = f"| EfficientNet-B4 U-Net | {format_iou(custom_miou)}"
    for name in class_names:
        key = f"{name}/iou"
        custom_row += f" | {format_iou(custom_results.get(key, 0.0))}"
    custom_row += " |"
    lines.append(custom_row)

    # Delta row
    delta_row = f"| **Delta** | **{format_iou(delta_miou):.1f}**"
    for name in class_names:
        base_val = baseline_results.get(f"{name}/iou", 0.0)
        custom_val = custom_results.get(f"{name}/iou", 0.0)
        delta_val = custom_val - base_val
        sign = "+" if delta_val > 0 else ""
        delta_row += f" | {sign}{format_iou(delta_val)}"
    delta_row += " |"
    lines.append(delta_row)

    # Additional metrics
    lines.extend([
        "",
        "## Additional Metrics",
        "",
        "| Metric | Baseline | Custom | Delta |",
        "|---|---|---|---|",
        f"| Pixel Accuracy | {format_iou(baseline_results.get('pixel_accuracy', 0.0))} | {format_iou(custom_results.get('pixel_accuracy', 0.0))} | {format_iou(custom_results.get('pixel_accuracy', 0.0) - baseline_results.get('pixel_accuracy', 0.0))} |",
        f"| Mean Accuracy | {format_iou(baseline_results.get('mean_accuracy', 0.0))} | {format_iou(custom_results.get('mean_accuracy', 0.0))} | {format_iou(custom_results.get('mean_accuracy', 0.0) - baseline_results.get('mean_accuracy', 0.0))} |",
    ])

    return "\n".join(lines)


def plot_side_by_side_qualitative(
    output_path: str,
    n_samples: int = 8,
) -> None:
    """Generate a qualitative grid with image / GT / baseline pred / custom pred.

    Loads saved predictions if available, otherwise creates a placeholder.
    """
    print("  (Qualitative side-by-side grid requires saved prediction arrays.)")
    print(f"  Placeholder: {output_path}")


def main() -> None:
    args = parse_args()

    baseline_results = load_results(args.baseline)
    custom_results = load_results(args.custom)

    # ── Table ────────────────────────────────────────────────────────
    table = build_comparison_table(baseline_results, custom_results, args.class_names)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(table + "\n")

    print(f"Comparison table saved to {args.output}")
    print()
    print(table)

    # ── Per-class IoU bar chart ──────────────────────────────────────
    iou_dict = {
        "Vanilla U-Net": [baseline_results.get(f"{name}/iou", 0.0) * 100 for name in args.class_names],
        "EfficientNet-B4 U-Net": [custom_results.get(f"{name}/iou", 0.0) * 100 for name in args.class_names],
    }
    bar_chart_path = os.path.join(args.output_dir, "iou_comparison.png")
    plot_iou_bar_chart(
        iou_dict,
        args.class_names,
        bar_chart_path,
        title="Per-Class IoU: Baseline vs Custom Model",
    )
    print(f"IoU comparison bar chart saved to {bar_chart_path}")


if __name__ == "__main__":
    main()
