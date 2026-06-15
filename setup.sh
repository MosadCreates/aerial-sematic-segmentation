#!/usr/bin/env bash
set -euo pipefail

# =========================================================================== #
# setup.sh — Full environment setup for Semantic Segmentation on Aerial Imagery
# =========================================================================== #
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What this does:
#   1. Checks Python 3.11+
#   2. Creates a virtual environment at .venv/
#   3. Installs PyTorch 2.1.2 with CUDA 12.1
#   4. Installs remaining pip dependencies
#   5. Creates directory structure
#   6. Downloads LoveDA dataset from HuggingFace
#   7. Verifies CUDA is available
# =========================================================================== #

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------------ #
# 1. Check Python version
# ------------------------------------------------------------------ #
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    log_error "Python not found. Please install Python 3.11+."
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    log_error "Python 3.11+ required. Found: $PY_VERSION"
    exit 1
fi
log_info "Python $PY_VERSION found."

# ------------------------------------------------------------------ #
# 2. Create virtual environment
# ------------------------------------------------------------------ #
if [ -d ".venv" ]; then
    log_warn ".venv already exists. Remove it first or run 'rm -rf .venv'."
    read -rp "Recreate .venv? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        rm -rf .venv
    else
        log_info "Using existing .venv."
    fi
fi

if [ ! -d ".venv" ]; then
    log_info "Creating .venv..."
    $PYTHON -m venv .venv
fi

# Activate (handle both bash/zsh and Windows Git Bash)
if [ -f ".venv/Scripts/activate" ]; then
    # Windows
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    # Unix
    source .venv/bin/activate
fi

log_info "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------ #
# 3. Install PyTorch with CUDA 12.1
# ------------------------------------------------------------------ #
log_info "Installing PyTorch 2.1.2 with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------------ #
# 4. Install remaining dependencies
# ------------------------------------------------------------------ #
log_info "Installing project dependencies..."
pip install -r requirements.txt

# ------------------------------------------------------------------ #
# 5. Create directory structure
# ------------------------------------------------------------------ #
log_info "Creating directory structure..."
mkdir -p data/loveda
mkdir -p checkpoints
mkdir -p logs
mkdir -p results/figures
touch src/utils/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/losses/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/serving/__init__.py

# ------------------------------------------------------------------ #
# 6. Download LoveDA dataset from HuggingFace
# ------------------------------------------------------------------ #
log_info "Downloading LoveDA dataset from HuggingFace..."
python -c "
from huggingface_hub import snapshot_download
import os

repo_id = 'tacoperis/loveda'
data_root = os.environ.get('DATA_ROOT', './data')
local_dir = os.path.join(data_root, 'loveda')

os.makedirs(local_dir, exist_ok=True)

print(f'Downloading {repo_id} to {local_dir}...')
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    repo_type='dataset',
    ignore_patterns=['*.md', '*.txt'],
)
print('LoveDA dataset downloaded successfully.')
"

# ------------------------------------------------------------------ #
# 7. Verify CUDA
# ------------------------------------------------------------------ #
log_info "Verifying CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

log_info "Setup complete!"
log_info "Activate the environment with:"
echo "  source .venv/bin/activate   # Linux/Mac"
echo "  source .venv/Scripts/activate  # Windows Git Bash"
