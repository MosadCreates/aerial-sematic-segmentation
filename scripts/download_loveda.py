"""
download_loveda.py — Download the LoveDA dataset from HuggingFace.

Usage:
    python scripts/download_loveda.py --data_root ./data

The LoveDA dataset is downloaded from the HuggingFace dataset hub
at tacoperis/loveda. The resulting folder structure is:

    data/loveda/
    ├── train/
    │   ├── Urban/
    │   │   ├── images_png/     (1024×1024 RGB images)
    │   │   └── masks_png/      (1024×1024 single-channel masks, values 0-6)
    │   └── Rural/
    │       ├── images_png/
    │       └── masks_png/
    ├── val/
    │   ├── Urban/
    │   └── Rural/
    └── test/
        ├── Urban/
        └── Rural/
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LoveDA dataset from HuggingFace"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for dataset storage (default: ./data)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="tacoperis/loveda",
        help="HuggingFace dataset repository ID (default: tacoperis/loveda)",
    )
    return parser.parse_args()


def download_loveda(data_root: str, repo_id: str) -> Path:
    local_dir = Path(data_root) / "loveda"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} to {local_dir}...")
    print("This may take a while (~5 GB). The dataset contains:")
    print("  - Train: Urban (1653 images) + Rural (1653 images) = 3306 images")
    print("  - Val:   Urban (382 images)  + Rural (382 images)  = 764 images")
    print("  - Test:  Urban (492 images)  + Rural (492 images)  = 984 images")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        repo_type="dataset",
        ignore_patterns=["*.md", "*.txt", "*.pdf", "*.yml", "*.yaml"],
    )

    # Verify expected structure
    splits = ["train", "val", "test"]
    domains = ["Urban", "Rural"]
    for split in splits:
        for domain in domains:
            img_dir = local_dir / split / domain / "images_png"
            if split != "test":
                mask_dir = local_dir / split / domain / "masks_png"
                assert mask_dir.exists(), f"Missing masks directory: {mask_dir}"
            if not img_dir.exists():
                print(f"  Warning: {img_dir} not found (may be expected for some splits)")

    # Count images
    total = 0
    for split in splits:
        for domain in domains:
            img_dir = local_dir / split / domain / "images_png"
            if img_dir.exists():
                count = len(list(img_dir.glob("*.png")))
                total += count
                print(f"  {split}/{domain}: {count} images")

    print(f"\nTotal images: {total}")
    print(f"LoveDA dataset downloaded to: {local_dir}")
    return local_dir


if __name__ == "__main__":
    args = parse_args()
    download_loveda(args.data_root, args.repo_id)
