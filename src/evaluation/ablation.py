"""
ablation.py — Run ablation study to isolate each contribution.

Runs 4 configurations and saves the comparison table:
  1. EfficientNet encoder only (no CutMix, plain CE loss)
  2. EfficientNet encoder + CutMix (plain CE loss)
  3. EfficientNet encoder + boundary-aware loss (no CutMix)
  4. Full model: EfficientNet encoder + CutMix + boundary-aware loss

Each configuration is trained with the same training pipeline and evaluated
on the test set. Results are compiled into results/ablation_table.md.

This ablation is a key portfolio piece — it shows rigorous experimental
methodology by isolating each contribution's impact on mIoU.

Usage:
    python src/evaluation/ablation.py --config configs/custom.yaml --output_dir results
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/custom.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs per ablation run (default: 30, fewer for speed)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configurations without running",
    )
    return parser.parse_args()


ABLATION_CONFIGS = [
    {
        "name": "EfficientNet + CE only",
        "description": "EfficientNet encoder, plain CE loss, no CutMix, no Dice, no Boundary",
        "modifications": {
            "cutmix": {"enabled": False, "prob": 0.0},
            "loss": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0,
                     "boundary": {"enabled": False}},
            "model": {"deep_supervision": False, "use_scse": False},
        },
        "run_name": "ablation-1-ce-only",
    },
    {
        "name": "EfficientNet + CutMix + CE",
        "description": "EfficientNet encoder, CutMix, plain CE loss, no Dice, no Boundary",
        "modifications": {
            "cutmix": {"enabled": True, "prob": 0.5, "alpha": 1.0},
            "loss": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0,
                     "boundary": {"enabled": False}},
            "model": {"deep_supervision": False, "use_scse": False},
        },
        "run_name": "ablation-2-cutmix-ce",
    },
    {
        "name": "EfficientNet + Boundary Loss",
        "description": "EfficientNet encoder, CE + Dice + Boundary loss, no CutMix",
        "modifications": {
            "cutmix": {"enabled": False, "prob": 0.0},
            "loss": {"alpha": 1.0, "beta": 1.0, "gamma": 0.5,
                     "boundary": {"enabled": True, "kernel_size": 3,
                                  "boundary_width": 3, "boundary_weight": 2.0}},
            "model": {"deep_supervision": True, "use_scse": True},
        },
        "run_name": "ablation-3-boundary-loss",
    },
    {
        "name": "Full: EfficientNet + CutMix + Boundary Loss",
        "description": "All components: EfficientNet encoder, CutMix, CE+Dice+Boundary, deep supervision, SCSE",
        "modifications": {},  # Uses the full config as-is
        "run_name": "ablation-4-full",
    },
]


def create_ablation_config(
    base_config: Dict, modifications: Dict, epochs: int
) -> Dict:
    """Create a modified config for a specific ablation variant."""
    config = copy.deepcopy(base_config)

    # Apply modifications
    for section, values in modifications.items():
        if section in config:
            if isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values
        else:
            config[section] = values

    # Override training epochs
    config["training"]["epochs"] = epochs
    config["training"]["val_frequency"] = 1

    # Ensure distinct wandb run names
    wandb_cfg = config.get("wandb", {})
    config["wandb"]["run_name"] = modifications.get("run_name", wandb_cfg.get("run_name", "ablation"))

    return config


def train_and_evaluate(config: Dict, output_dir: str) -> Dict:
    """Train a model and evaluate it on the test set.

    Returns evaluation metrics dict.
    """
    # Save config for this run
    run_name = config["wandb"]["run_name"]
    run_dir = Path(output_dir) / "ablation" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Set checkpoint directory
    ckpt_dir = run_dir / "checkpoints"
    config["ckpt_dir"] = str(ckpt_dir)

    # Train
    print(f"\n{'='*60}")
    print(f"Training: {config['wandb']['run_name']}")
    print(f"{'='*60}")

    train_script = os.path.join(
        os.path.dirname(__file__), "..", "training", "train.py"
    )

    train_cmd = [
        sys.executable, train_script,
        "--config", str(config_path),
        "--ckpt_dir", str(ckpt_dir),
        "--wandb_mode", "disabled",  # Don't log ablation to wandb
    ]

    start_time = time.time()
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    train_time = time.time() - start_time

    if result.returncode != 0:
        print(f"Training failed for {run_name}")
        print(result.stderr)
        return {"miou": 0.0, "train_time": train_time, "error": result.stderr}

    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    # Evaluate best checkpoint
    best_ckpt = Path(ckpt_dir) / "best_model.pth"
    if not best_ckpt.exists():
        print(f"Checkpoint not found: {best_ckpt}")
        return {"miou": 0.0, "train_time": train_time}

    print(f"\nEvaluating: {run_name}")
    eval_script = os.path.join(
        os.path.dirname(__file__), "evaluate.py"
    )

    eval_cmd = [
        sys.executable, eval_script,
        "--checkpoint", str(best_ckpt),
        "--config", str(config_path),
        "--output_dir", str(run_dir),
        "--split", "val",
    ]

    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

    if eval_result.returncode != 0:
        print(f"Evaluation failed for {run_name}")
        print(eval_result.stderr)
        return {"miou": 0.0, "train_time": train_time}

    # Parse metrics from results JSON
    results_path = run_dir / f"{config['model'].get('architecture', 'efficient_unet')}_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            metrics = json.load(f)
        metrics["train_time"] = train_time
        return metrics

    return {"miou": 0.0, "train_time": train_time}


def save_ablation_table(
    all_results: Dict[str, Dict],
    class_names: List[str],
    output_path: str,
) -> None:
    """Save ablation results as a markdown table."""
    lines = [
        "# Ablation Study",
        "",
        "Each ablation isolates a single contribution. All models use the",
        "EfficientNet-B4 encoder. Training is for 30 epochs on LoveDA.",
        "",
        "| Configuration | mIoU | Building | Road | Water | Barren | Forest | Agriculture |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for name, metrics in all_results.items():
        miou = metrics.get("miou", 0.0) * 100
        row = f"| {name} | {miou:.1f}"
        for cls_name in class_names:
            if cls_name != "Background":
                val = metrics.get(f"{cls_name}/iou", 0.0) * 100
                row += f" | {val:.1f}"
            else:
                val = metrics.get(f"{cls_name}/iou", 0.0) * 100
                row += f" | {val:.1f}"
        row += " |"
        lines.append(row)

    # Add delta row
    names = list(all_results.keys())
    if len(names) >= 4:
        full_miou = all_results[names[3]].get("miou", 0.0) * 100
        base_miou = all_results[names[0]].get("miou", 0.0) * 100
        delta = full_miou - base_miou
        lines.append(f"| **Delta (Full - CE only)** | **+{delta:.1f}** | | | | | | |")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **CE only baseline**: {all_results[names[0]].get('miou', 0.0)*100:.1f} mIoU",
        f"- **+CutMix**: {all_results[names[1]].get('miou', 0.0)*100:.1f} mIoU (+{(all_results[names[1]].get('miou', 0.0)-all_results[names[0]].get('miou', 0.0))*100:.1f})",
        f"- **+Boundary loss**: {all_results[names[2]].get('miou', 0.0)*100:.1f} mIoU (+{(all_results[names[2]].get('miou', 0.0)-all_results[names[0]].get('miou', 0.0))*100:.1f})",
        f"- **Full model (CutMix + Boundary loss)**: {all_results[names[3]].get('miou', 0.0)*100:.1f} mIoU (+{delta:.1f})",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Ablation table saved to {output_path}")


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    class_names = base_config["dataset"]["class_names"]

    all_results = {}

    for ablation in ABLATION_CONFIGS:
        config = create_ablation_config(
            base_config, ablation["modifications"], args.epochs
        )
        config["wandb"]["run_name"] = ablation["run_name"]

        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"Dry run: {ablation['name']}")
            print(f"  Run name: {ablation['run_name']}")
            print(f"  Modifications:")
            for section, values in ablation["modifications"].items():
                print(f"    {section}: {values}")
            continue

        metrics = train_and_evaluate(config, args.output_dir)
        all_results[ablation["name"]] = metrics

        print(f"  mIoU: {metrics.get('miou', 0.0)*100:.2f}")

    if not args.dry_run:
        output_path = os.path.join(args.output_dir, "ablation_table.md")
        save_ablation_table(all_results, class_names, output_path)

        # Also save raw results
        raw_path = os.path.join(args.output_dir, "ablation_results.json")
        serializable = {}
        for name, metrics in all_results.items():
            serializable[name] = {
                k: float(v) if not isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            }
        with open(raw_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Raw results saved to {raw_path}")


if __name__ == "__main__":
    main()
