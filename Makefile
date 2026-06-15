.PHONY: setup data baseline train ablation eval export benchmark serve docker-build test lint clean help

# =========================================================================== #
# Makefile — Semantic Segmentation on Aerial Imagery
# =========================================================================== #
# Usage:
#   make setup       # Full setup: venv + deps + dataset
#   make data        # Compute dataset stats (class weights, mean/std, samples)
#   make baseline    # Train vanilla U-Net baseline
#   make train       # Train custom EfficientNet-B4 U-Net
#   make ablation    # Run 4-variant ablation study
#   make eval        # Evaluate trained model on test set
#   make export      # Export trained model to ONNX
#   make benchmark   # Benchmark inference latency
#   make serve       # Start FastAPI inference server
#   make docker-build# Build Docker image
#   make test        # Run unit tests
#   make lint        # Check code formatting
#   make clean       # Clean generated files
# =========================================================================== #

# ── Default target ──────────────────────────────────────────────────────────
help:
	@echo "Usage:"
	@echo "  make setup       Full environment setup (venv + deps + dataset)"
	@echo "  make data        Compute dataset statistics & class weights"
	@echo "  make baseline    Train vanilla U-Net baseline"
	@echo "  make train       Train custom EfficientNet-B4 U-Net"
	@echo "  make ablation    Run 4-variant ablation study (30 epochs each)"
	@echo "  make eval        Evaluate trained model on test set"
	@echo "  make export      Export trained model to ONNX"
	@echo "  make benchmark   Benchmark inference latency (PyTorch FP32/FP16 + ONNX)"
	@echo "  make serve       Start FastAPI inference server (port 8000)"
	@echo "  make docker-build Build Docker image for inference server"
	@echo "  make test        Run unit tests"
	@echo "  make lint        Check code formatting (ruff + black)"
	@echo "  make clean       Remove generated files and caches"

# ── Configuration ──────────────────────────────────────────────────────────
PYTHON = python
CONFIG_DIR = configs
CKPT_DIR = checkpoints
RESULTS_DIR = results

# ── Targets ─────────────────────────────────────────────────────────────────
setup:
	@echo "=== Running full setup ==="
	@chmod +x setup.sh
	@./setup.sh

data:
	@echo "=== Computing dataset statistics ==="
	$(PYTHON) src/data/dataset_stats.py --config $(CONFIG_DIR)/config.yaml

baseline:
	@echo "=== Training baseline model ==="
	$(PYTHON) src/training/train.py --config $(CONFIG_DIR)/baseline.yaml

train:
	@echo "=== Training custom model ==="
	$(PYTHON) src/training/train.py --config $(CONFIG_DIR)/custom.yaml

ablation:
	@echo "=== Running ablation study ==="
	$(PYTHON) src/evaluation/ablation.py --config $(CONFIG_DIR)/custom.yaml --epochs 30

compare:
	@echo "=== Generating comparison table ==="
	$(PYTHON) src/evaluation/comparison.py \
		--baseline results/baseline/vanilla_unet_results.json \
		--custom results/custom/efficient_unet_results.json \
		--output results/comparison_table.md

eval:
	@echo "=== Evaluating model ==="
	@read -p "Checkpoint path: " ckpt; \
	$(PYTHON) src/evaluation/evaluate.py --checkpoint $$ckpt --config $(CONFIG_DIR)/config.yaml

export:
	@echo "=== Exporting model to ONNX ==="
	@read -p "Checkpoint path: " ckpt; \
	$(PYTHON) src/serving/export_onnx.py --checkpoint $$ckpt --config $(CONFIG_DIR)/config.yaml

benchmark:
	@echo "=== Running inference benchmark ==="
	@read -p "Checkpoint path: " ckpt; \
	$(PYTHON) src/serving/benchmark.py --checkpoint $$ckpt --config $(CONFIG_DIR)/config.yaml --output results/inference_benchmark.json

serve:
	@echo "=== Starting inference server ==="
	$(PYTHON) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	@echo "=== Building Docker image ==="
	bash scripts/docker_build.sh

test:
	@echo "=== Running tests ==="
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	@echo "=== Running linter ==="
	ruff check src/
	black --check src/

clean:
	@echo "=== Cleaning ==="
	rm -rf __pycache__ .pytest_cache
	rm -rf src/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."
