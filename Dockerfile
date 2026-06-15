# =========================================================================== #
# Dockerfile — Multi-stage build for FastAPI inference server
# =========================================================================== #
# Stage 1: Base image with dependencies
# Stage 2: Runtime image with model and server
#
# Build:
#   docker build -t aerial-segmentation:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/checkpoints:/app/checkpoints \
#     -e CHECKPOINT_PATH=/app/checkpoints/best_model.pth \
#     aerial-segmentation:latest
# =========================================================================== #

# ── Stage 1: Build dependencies ──────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Stage 2: Runtime image ──────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CHECKPOINT_PATH=/app/checkpoints/best_model.pth \
    CONFIG_PATH=/app/configs/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy Python packages from build stage
COPY --from=build /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start server
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
