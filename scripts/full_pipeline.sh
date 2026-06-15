#!/usr/bin/env bash
# =========================================================================== #
# full_pipeline.sh — End-to-end pipeline runner
# =========================================================================== #
# Runs the complete pipeline from setup through evaluation and export.
#
# Usage:
#   bash scripts/full_pipeline.sh
#
# This script will:
#   1. Set up the environment
#   2. Download the LoveDA dataset
#   3. Compute dataset statistics
#   4. Train the vanilla U-Net baseline
#   5. Train the custom EfficientNet-B4 U-Net
#   6. Run ablation study
#   7. Generate comparison table
#   8. Export model to ONNX
#   9. Begin benchmark (requires GPU)
#
# Prerequisites:
#   - Python 3.11+
#   - CUDA-capable GPU (A100 40GB recommended)
#   - 20GB+ free disk space for dataset and checkpoints
# =========================================================================== #

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

log_info "Project root: $PROJECT_DIR"
log_info "Starting full pipeline..."
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Environment setup
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 1/9: Setting up environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
else
    source .venv/bin/activate
    log_info "  Existing .venv found, activating."
fi

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); exit(0 if torch.cuda.is_available() else 1)" || {
    log_warn "  CUDA not available. Training will be slow or impossible on CPU."
}

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Dataset download
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 2/9: Downloading LoveDA dataset..."
python scripts/download_loveda.py --data_root ./data
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Dataset statistics
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 3/9: Computing dataset statistics..."
python src/data/dataset_stats.py --config configs/config.yaml --output_dir results
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Train baseline
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 4/9: Training vanilla U-Net baseline..."
python src/training/train.py --config configs/baseline.yaml --ckpt_dir checkpoints/baseline
BASELINE_CKPT="checkpoints/baseline/best_model.pth"
if [ -f "$BASELINE_CKPT" ]; then
    log_info "  Baseline checkpoint: $BASELINE_CKPT"
fi
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Evaluate baseline
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 5/9: Evaluating baseline..."
python src/evaluation/evaluate.py \
    --checkpoint "$BASELINE_CKPT" \
    --config configs/baseline.yaml \
    --output_dir results/baseline \
    --split val
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Train custom model
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 6/9: Training EfficientNet-B4 U-Net..."
python src/training/train.py --config configs/custom.yaml --ckpt_dir checkpoints/custom
CUSTOM_CKPT="checkpoints/custom/best_model.pth"
if [ -f "$CUSTOM_CKPT" ]; then
    log_info "  Custom checkpoint: $CUSTOM_CKPT"
fi
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Evaluate custom model
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 7/9: Evaluating custom model..."
python src/evaluation/evaluate.py \
    --checkpoint "$CUSTOM_CKPT" \
    --config configs/custom.yaml \
    --output_dir results/custom \
    --split val

# Generate comparison table
if [ -f "results/baseline/vanilla_unet_results.json" ] && [ -f "results/custom/efficient_unet_results.json" ]; then
    python src/evaluation/comparison.py \
        --baseline results/baseline/vanilla_unet_results.json \
        --custom results/custom/efficient_unet_results.json \
        --output results/comparison_table.md
fi
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 8: Run ablation
# ──────────────────────────────────────────────────────────────────────────────
log_info "Step 8/9: Running ablation study (4 variants, 30 epochs each)..."
python src/evaluation/ablation.py --config configs/custom.yaml --epochs 30 --output_dir results
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 9: Export to ONNX
# ──────────────────────────────────────────────────────────────────────────────
if [ -f "$CUSTOM_CKPT" ]; then
    log_info "Step 9/9: Exporting custom model to ONNX..."
    python src/serving/export_onnx.py \
        --checkpoint "$CUSTOM_CKPT" \
        --config configs/custom.yaml \
        --output models/custom.onnx \
        --verify
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
log_info "Pipeline complete!"
echo ""
echo "Results summary:"
echo "  ─────────────────────────────────────────────"
echo "  Baseline checkpoint:  $BASELINE_CKPT"
echo "  Custom checkpoint:    $CUSTOM_CKPT"
echo "  ONNX model:           models/custom.onnx"
echo "  Results:              results/"
echo "  Comparison table:     results/comparison_table.md"
echo "  Ablation table:       results/ablation_table.md"
echo "  ─────────────────────────────────────────────"
echo ""
log_info "To start the inference server:"
echo "  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000"
echo ""
log_info "To run tests:"
echo "  pytest tests/ -v"
