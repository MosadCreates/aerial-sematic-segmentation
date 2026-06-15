"""
train.py — Full training pipeline for baseline and custom models.

Supports both:
  - Vanilla U-Net baseline (--config configs/baseline.yaml)
  - EfficientNet-B4 U-Net (--config configs/custom.yaml)

Key features:
  - Mixed-precision training via torch.cuda.amp
  - CutMix augmentation at batch level
  - Deep supervision loss with auxiliary weight annealing
  - Gradient clipping and accumulation
  - Cosine annealing with warm restarts + linear warmup
  - Differential learning rates (encoder vs decoder)
  - Sliding window validation
  - wandb logging with segmentation overlays
  - Checkpointing by best val mIoU

Mixed-precision explanation:
  Operations that benefit from FP16:
    - Convolutions (compute-bound, 2x throughput in FP16)
    - Matrix multiplications (attention, FC layers)
    - Non-linear activations (ReLU, sigmoid)
  Operations that must stay FP32:
    - BatchNorm (requires FP32 statistics)
    - Loss scaling (GradScaler handles this)
    - Softmax (numerical stability near 0)
  autocast handles this automatically: FP16 for conv/linear, FP32 for BN/softmax.

Gradient accumulation:
  When batch_size is limited by VRAM (e.g., 8 on A100), gradient accumulation
  simulates a larger effective batch size by accumulating gradients over
  multiple forward-backward passes before stepping the optimiser.
  Effective batch size = batch_size × accumulation_steps.

Gradient clipping:
  Important with deep supervision because multiple loss terms could produce
  large combined gradients. Clipping at max_norm=1.0 prevents exploding gradients.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.augmentation import get_transforms
from src.data.class_weights import get_loveda_class_weights
from src.data.cutmix import CutMix
from src.data.dataset import LoveDA
from src.evaluation.metrics import ConfusionMatrix, compute_all_metrics
from src.losses.composite_loss import BoundaryAwareLoss
from src.models.efficient_unet import EfficientUNet
from src.models.vanilla_unet import VanillaUNet
from src.training.sliding_window import sliding_window_inference
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.visualization import (
    log_wandb_images,
    plot_iou_bar_chart,
    plot_qualitative_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train segmentation model on LoveDA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (overrides config)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Override wandb mode",
    )
    return parser.parse_args()


def build_model(config: Dict, num_classes: int) -> nn.Module:
    """Build model from config."""
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "efficient_unet")

    if arch == "vanilla_unet":
        model = VanillaUNet(num_classes=num_classes)
    elif arch == "efficient_unet":
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
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model


def build_optimizer(
    model: nn.Module, config: Dict
) -> torch.optim.Optimizer:
    """Build optimiser with differential learning rates for encoder/decoder."""
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "efficient_unet")

    if arch == "vanilla_unet":
        return AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
        )

    # Differential learning rates for encoder vs decoder
    encoder_params = model.get_encoder_params()
    decoder_params = model.get_decoder_params()

    encoder_lr = opt_cfg.get("encoder_lr", opt_cfg["lr"] * 0.1)

    param_groups = [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": decoder_params, "lr": opt_cfg["lr"]},
    ]

    return AdamW(
        param_groups,
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    steps_per_epoch: int,
) -> Dict:
    """Build learning rate scheduler with warmup.

    Returns dict with scheduler and warmup_scheduler.
    """
    sched_cfg = config["scheduler"]
    warmup_epochs = sched_cfg.get("warmup_epochs", 0)

    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get("t_0", 10),
        T_mult=sched_cfg.get("t_mult", 2),
        eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    # Linear warmup
    warmup_scheduler = None
    if warmup_epochs > 0:

        def warmup_fn(epoch_fraction):
            warmup_steps = warmup_epochs * steps_per_epoch
            step = epoch_fraction * steps_per_epoch
            if step < warmup_steps:
                return (step / warmup_steps) * (
                    sched_cfg.get("warmup_lr", 1e-6) / sched_cfg.get("eta_min", 1e-6)
                ) + (1.0 - step / warmup_steps) * 0.1
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)

    return {"scheduler": scheduler, "warmup_scheduler": warmup_scheduler}


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    config: Dict,
    cutmix: Optional[CutMix] = None,
    scheduler_dict: Optional[Dict] = None,
    device: torch.device = torch.device("cuda"),
) -> Dict:
    """Run one training epoch.

    Returns:
        Dict with loss, learning rate, gradient norm, and throughput metrics.
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    grad_norm_steps = 0
    num_batches = len(dataloader)
    train_cfg = config["training"]
    accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    clip_norm = train_cfg.get("gradient_clip_norm", 1.0)
    log_freq = train_cfg.get("log_frequency", 50)
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "efficient_unet")

    epoch_start = time.time()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images, masks = batch[0].to(device), batch[1].to(device)

        # CutMix augmentation
        if cutmix is not None and arch == "efficient_unet":
            images, masks = cutmix(images, masks)

        # Mixed-precision forward
        with autocast(enabled=train_cfg.get("mixed_precision", True)):
            outputs = model(images)
            loss = criterion(outputs, masks, current_epoch=epoch)

        # Normalise for gradient accumulation
        loss = loss / accumulation_steps

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            total_grad_norm += float(grad_norm)
            grad_norm_steps += 1

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update learning rate (warmup then cosine)
            if scheduler_dict:
                if (
                    scheduler_dict["warmup_scheduler"]
                    and epoch * num_batches + batch_idx
                    < config["scheduler"]["warmup_epochs"] * num_batches
                ):
                    scheduler_dict["warmup_scheduler"].step()
                else:
                    scheduler_dict["scheduler"].step()

        total_loss += loss.item() * accumulation_steps

        if batch_idx % log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item()*accumulation_steps:.4f}", "lr": f"{lr:.2e}"})

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches
    lr = optimizer.param_groups[0]["lr"]
    avg_grad_norm = total_grad_norm / max(grad_norm_steps, 1)
    images_per_sec = len(dataloader.dataset) / max(epoch_time, 1e-8)

    return {
        "train_loss": avg_loss,
        "learning_rate": lr,
        "gradient_norm": avg_grad_norm,
        "throughput": images_per_sec,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataset: LoveDA,
    config: Dict,
    device: torch.device = torch.device("cuda"),
) -> Dict:
    """Run validation with sliding window inference.

    Args:
        model: Trained model.
        dataset: LoveDA validation dataset (transforms applied).
        config: Full config dict.
        device: Device.

    Returns:
        Dict with all validation metrics.
    """
    model.eval()
    sw_cfg = config["sliding_window"]
    patch_size = sw_cfg.get("patch_size", 512)
    stride = sw_cfg.get("stride", 256)
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["class_names"]

    cm = ConfusionMatrix(num_classes)

    for idx in tqdm(range(len(dataset)), desc="Validation"):
        image, mask = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)

        pred = sliding_window_inference(
            model=model,
            image=image,
            patch_size=patch_size,
            stride=stride,
            num_classes=num_classes,
            gaussian_weight=sw_cfg.get("gaussian_weight", True),
            device=device,
            mixed_precision=config["training"].get("mixed_precision", True),
        )

        cm.update(pred, mask.numpy())

    metrics = compute_all_metrics(cm, class_names)
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_dict: Dict,
    epoch: int,
    metrics: Dict,
    config: Dict,
    is_best: bool = False,
) -> None:
    """Save training checkpoint."""
    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    if is_best:
        path = ckpt_dir / "best_model.pth"
    else:
        path = ckpt_dir / "last_model.pth"

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def plot_lr_schedule(
    config: Dict,
    steps_per_epoch: int,
    output_path: str = "results/lr_schedule.png",
) -> None:
    """Simulate and plot the complete LR schedule over all epochs."""
    epochs = config["training"]["epochs"]
    sched_cfg = config["scheduler"]
    warmup_epochs = sched_cfg.get("warmup_epochs", 0)
    base_lr = config["optimizer"]["lr"]

    total_steps = epochs * steps_per_epoch
    lrs = []

    optimizer = AdamW([torch.nn.Parameter(torch.randn(1))], lr=base_lr)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get("t_0", 10),
        T_mult=sched_cfg.get("t_mult", 2),
        eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    for step in range(total_steps):
        epoch = step / steps_per_epoch
        if warmup_epochs > 0 and epoch < warmup_epochs:
            progress = epoch / warmup_epochs
            lr = base_lr * (0.1 + 0.9 * progress)
        else:
            scheduler.step(epoch)
            lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lrs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"LR Schedule: {sched_cfg.get('name', 'cosine_warm_restarts')}")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"LR schedule plot saved to {output_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load checkpoint and return starting epoch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 0)})")
    return start_epoch


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Override config with CLI args
    if args.data_root:
        config["data_root"] = args.data_root
    if args.ckpt_dir:
        config["ckpt_dir"] = args.ckpt_dir
    if args.resume:
        config["training"]["resume"] = args.resume
    if args.wandb_mode:
        config["wandb"]["mode"] = args.wandb_mode

    # Set seed
    set_seed(config.get("seed", 42))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Dataset parameters
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["class_names"]
    data_root = config["data_root"]

    # Class weights
    freq_path = os.path.join(config.get("results_dir", "results"), "dataset_stats.yaml")
    if os.path.exists(freq_path):
        with open(freq_path, "r") as f:
            stats = yaml.safe_load(f)
        class_weights = torch.tensor(stats.get("class_weights", [1.0] * num_classes))
    else:
        class_weights = None
        print("Warning: dataset_stats.yaml not found. Using uniform class weights.")
    print(f"Class weights: {class_weights}")

    # Build model
    model = build_model(config, num_classes)
    model = model.to(device)
    print(f"Model: {config['model'].get('architecture', 'efficient_unet')}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable:,}")

    # Build loss
    criterion = BoundaryAwareLoss(config, class_weights=class_weights, num_classes=num_classes)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)

    # Build dataloaders
    train_transform = get_transforms("train", config)
    val_transform = get_transforms("val", config)

    train_dataset = LoveDA(
        root=data_root, split="train", domains=["Urban", "Rural"], transform=train_transform
    )
    val_dataset = LoveDA(
        root=data_root, split="val", domains=["Urban", "Rural"], transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Scheduler
    scheduler_dict = build_scheduler(optimizer, config, len(train_loader))

    # CutMix
    cutmix_config = config.get("cutmix", {})
    cutmix = CutMix(
        alpha=cutmix_config.get("alpha", 1.0),
        prob=cutmix_config.get("prob", 0.5) if cutmix_config.get("enabled", True) else 0.0,
    )

    # Mixed precision
    scaler = GradScaler(enabled=config["training"].get("mixed_precision", True))

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0
    resume_path = config["training"].get("resume", None)
    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(resume_path, model, optimizer)
        best_miou = torch.load(resume_path, map_location="cpu").get("metrics", {}).get("miou", 0.0)

    # Freeze encoder if configured
    if hasattr(model, "freeze_encoder") and config["model"].get("encoder_freeze_epochs", 0) > 0:
        model.freeze_encoder()
        print(f"Encoder frozen for first {config['model']['encoder_freeze_epochs']} epochs")

    # wandb
    use_wandb = False
    if config["wandb"]["mode"] != "disabled":
        try:
            import wandb

            wandb.init(
                project=config["wandb"].get("project", "aerial-semantic-segmentation"),
                entity=config["wandb"].get("entity"),
                name=config["wandb"].get("run_name"),
                config=config,
                mode=config["wandb"]["mode"],
            )
            use_wandb = True
        except Exception as e:
            print(f"wandb init failed: {e}")

    # Training loop
    epochs = config["training"]["epochs"]
    val_freq = config["training"].get("val_frequency", 1)
    viz_freq = config["training"].get("visualize_frequency", 5)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Gradient accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    print(f"Effective batch size: {config['training']['batch_size'] * config['training'].get('gradient_accumulation_steps', 1)}")
    print(f"Mixed precision: {config['training'].get('mixed_precision', True)}")

    # Plot and save LR schedule
    results_dir = config.get("results_dir", "results")
    plot_lr_schedule(config, len(train_loader), os.path.join(results_dir, "lr_schedule.png"))

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Unfreeze encoder at specified epoch
        if (
            hasattr(model, "unfreeze_encoder")
            and epoch == config["model"].get("encoder_freeze_epochs", 0)
            and epoch > 0
        ):
            model.unfreeze_encoder()
            print(f"Encoder unfrozen at epoch {epoch}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            config=config,
            cutmix=cutmix,
            scheduler_dict=scheduler_dict,
            device=device,
        )

        epoch_time = time.time() - epoch_start

        # Validate
        val_metrics = {}
        if (epoch + 1) % val_freq == 0:
            val_metrics = validate(model, val_dataset, config, device)

            # Update best mIoU and save checkpoint
            current_miou = val_metrics.get("miou", 0.0)
            if current_miou > best_miou:
                best_miou = current_miou
                save_checkpoint(model, optimizer, scheduler_dict, epoch, val_metrics, config, is_best=True)

            # Log
            print(f"\nEpoch {epoch+1}/{epochs} | "
                  f"Loss: {train_metrics['train_loss']:.4f} | "
                  f"mIoU: {current_miou:.4f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"LR: {train_metrics['learning_rate']:.2e}")

            # GPU memory tracking
            gpu_mem_allocated = 0
            gpu_mem_cached = 0
            if device.type == "cuda":
                gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                gpu_mem_cached = torch.cuda.memory_reserved(device) / 1024**3

            if use_wandb:
                import wandb
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics["train_loss"],
                    "train/lr": train_metrics["learning_rate"],
                    "train/gradient_norm": train_metrics["gradient_norm"],
                    "train/throughput": train_metrics["throughput"],
                    "val/miou": val_metrics["miou"],
                    "val/pixel_accuracy": val_metrics["pixel_accuracy"],
                    "val/mean_accuracy": val_metrics["mean_accuracy"],
                }
                for name in class_names:
                    log_dict[f"val/iou/{name}"] = val_metrics.get(f"{name}/iou", 0.0)
                if device.type == "cuda":
                    log_dict["system/gpu_mem_allocated_gb"] = gpu_mem_allocated
                    log_dict["system/gpu_mem_cached_gb"] = gpu_mem_cached

                # Per-class IoU bar chart
                iou_values = {name: val_metrics.get(f"{name}/iou", 0.0) * 100 for name in class_names}
                log_dict["val/per_class_iou"] = wandb.Bar(
                    list(iou_values.keys()), list(iou_values.values()),
                    title="Per-Class IoU (%)"
                )

                wandb.log(log_dict)

        else:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {train_metrics['train_loss']:.4f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"LR: {train_metrics['learning_rate']:.2e}")

            gpu_mem_allocated = 0
            if device.type == "cuda":
                gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3

            if use_wandb:
                import wandb
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics["train_loss"],
                    "train/lr": train_metrics["learning_rate"],
                    "train/gradient_norm": train_metrics["gradient_norm"],
                    "train/throughput": train_metrics["throughput"],
                }
                if device.type == "cuda":
                    log_dict["system/gpu_mem_allocated_gb"] = gpu_mem_allocated
                wandb.log(log_dict)

        # Visualise every N epochs
        if (epoch + 1) % viz_freq == 0 and use_wandb and config["wandb"].get("log_images", True):
            try:
                images_list, gt_list, pred_list = [], [], []
                for i in range(min(4, len(val_dataset))):
                    img, mask = val_dataset[i]
                    pred = sliding_window_inference(
                        model, img, device=device,
                        **config["sliding_window"],
                        num_classes=num_classes,
                        mixed_precision=config["training"].get("mixed_precision", True),
                    )
                    images_list.append(img.permute(1, 2, 0).cpu().numpy())
                    gt_list.append(mask.numpy())
                    pred_list.append(pred)

                log_wandb_images(
                    wandb, images_list, gt_list, pred_list,
                    class_names, prefix=f"val_epoch_{epoch}",
                )
            except Exception as e:
                print(f"Visualisation logging failed: {e}")

        # Save last checkpoint
        save_checkpoint(model, optimizer, scheduler_dict, epoch, val_metrics or {"miou": 0}, config, is_best=False)

    # Final results
    print(f"\nTraining complete! Best val mIoU: {best_miou:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
