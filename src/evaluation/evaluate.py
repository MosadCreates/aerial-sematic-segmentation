"""
evaluate.py — Evaluation harness for trained models.

Loads a trained checkpoint and evaluates on the full test set using
sliding window inference. Computes per-class IoU, mIoU, pixel accuracy,
confusion matrix, and generates visualisations.

Usage:
    python src/evaluation/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/config.yaml
    python src/evaluation/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/baseline.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.augmentation import get_transforms
from src.data.dataset import LoveDA
from src.evaluation.metrics import (
    ConfusionMatrix,
    compute_all_metrics,
    compute_confusion_matrix,
)
from src.models.efficient_unet import EfficientUNet
from src.models.vanilla_unet import VanillaUNet
from src.training.sliding_window import sliding_window_inference
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_iou_bar_chart,
    plot_qualitative_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on LoveDA test set"
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
        "--data_root",
        type=str,
        default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str, config: Dict, device: torch.device
) -> torch.nn.Module:
    """Load model architecture and weights from checkpoint."""
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "efficient_unet")
    num_classes = config["dataset"]["num_classes"]

    if arch == "vanilla_unet":
        model = VanillaUNet(num_classes=num_classes)
    elif arch == "efficient_unet":
        model = EfficientUNet(
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b4"),
            encoder_weights=None,  # No pretrained weights needed for eval
            num_classes=num_classes,
            decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32, 16]),
            use_scse=model_cfg.get("use_scse", True),
            deep_supervision=False,  # Remove aux heads at inference
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Handle aux head keys if present in checkpoint but not in model
    if arch == "efficient_unet":
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("aux_heads.")}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Architecture: {arch}")
    print(f"Previous val mIoU: {checkpoint.get('metrics', {}).get('miou', 'N/A')}")

    return model


def evaluate(
    model: torch.nn.Module,
    dataset: LoveDA,
    config: Dict,
    device: torch.device = torch.device("cuda"),
) -> Dict:
    """Run full evaluation on a dataset.

    Args:
        model: Trained model.
        dataset: LoveDA dataset (val or test).
        config: Full config.
        device: Device.

    Returns:
        Dictionary with all predictions, ground truths, and images.
    """
    sw_cfg = config["sliding_window"]
    patch_size = sw_cfg.get("patch_size", 512)
    stride = sw_cfg.get("stride", 256)
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["class_names"]

    cm = ConfusionMatrix(num_classes)
    all_images = []
    all_gts = []
    all_preds = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        image, mask = dataset[idx]

        pred = sliding_window_inference(
            model=model,
            image=image,
            patch_size=patch_size,
            stride=stride,
            num_classes=num_classes,
            gaussian_weight=sw_cfg.get("gaussian_weight", True),
            device=device,
            mixed_precision=True,
        )

        cm.update(pred, mask.numpy())
        all_images.append(image.permute(1, 2, 0).cpu().numpy())
        all_gts.append(mask.numpy())
        all_preds.append(pred)

    metrics = compute_all_metrics(cm, class_names)

    return {"metrics": metrics, "images": all_images, "gts": all_gts, "preds": all_preds, "cm": cm}


def save_results(
    results: Dict,
    config: Dict,
    output_dir: str,
    model_name: str = "model",
) -> None:
    """Save evaluation results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = config["dataset"]["class_names"]
    metrics = results["metrics"]

    # ── JSON results ───────────────────────────────────────────────────
    results_path = output_dir / f"{model_name}_results.json"
    serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Print results table ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"  mIoU:              {metrics['miou']*100:.2f}")
    print(f"  Pixel accuracy:    {metrics['pixel_accuracy']*100:.2f}")
    print(f"  Mean accuracy:     {metrics['mean_accuracy']*100:.2f}")
    print(f"\n  Per-class IoU:")
    for name in class_names:
        print(f"    {name:15s}: {metrics[f'{name}/iou']*100:.2f}")
    print(f"{'='*60}")

    # ── Confusion matrix ───────────────────────────────────────────────
    num_classes = len(class_names)
    full_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            full_cm[i, j] = results["cm"].tp[j] if i == j else 0  # Simplified
    # Properly compute confusion matrix from predictions
    all_preds = np.array(results["preds"])
    all_gts = np.array(results["gts"])
    full_cm = compute_confusion_matrix(all_preds, all_gts, num_classes)

    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(full_cm, class_names, str(cm_path))
    print(f"Confusion matrix saved to {cm_path}")

    # ── Qualitative grid ───────────────────────────────────────────────
    qual_path = output_dir / f"{model_name}_qualitative.png"
    plot_qualitative_grid(
        results["images"],
        results["gts"],
        results["preds"],
        str(qual_path),
        n_samples=8,
        model_name=model_name,
    )
    print(f"Qualitative grid saved to {qual_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.data_root:
        config["data_root"] = args.data_root

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    # Load dataset
    val_transform = get_transforms("val", config)
    dataset = LoveDA(
        root=config["data_root"],
        split=args.split,
        domains=["Urban", "Rural"],
        transform=val_transform,
    )
    print(f"Dataset: {args.split} split, {len(dataset)} images")

    # Evaluate
    results = evaluate(model, dataset, config, device)

    # Save results
    model_name = config["model"].get("architecture", "model")
    save_results(results, config, args.output_dir, model_name)


if __name__ == "__main__":
    main()
