#!/usr/bin/env bash
# =========================================================================== #
# docker_build.sh — Docker build and push helper
# =========================================================================== #
# Usage:
#   bash scripts/docker_build.sh              # Build only
#   bash scripts/docker_build.sh push         # Build + push to registry
#
# Environment variables:
#   REGISTRY: Container registry (default: docker.io)
#   IMAGE_NAME: Image name (default: aerial-segmentation)
#   TAG: Image tag (default: latest)
# =========================================================================== #

set -euo pipefail

REGISTRY="${REGISTRY:-docker.io}"
IMAGE_NAME="${IMAGE_NAME:-aerial-segmentation}"
TAG="${TAG:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Building Docker image..."
echo "  Image:   $FULL_IMAGE"
echo "  Context: $PROJECT_DIR"
echo ""

# Build
docker build \
    -t "$FULL_IMAGE" \
    -f Dockerfile \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    .

echo ""
echo "Build complete: $FULL_IMAGE"
docker images "$FULL_IMAGE" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

# Optional: push
if [ "${1:-}" = "push" ]; then
    echo ""
    echo "Pushing to registry..."
    docker push "$FULL_IMAGE"
    echo "Push complete."
fi

echo ""
echo "To run:"
echo "  docker run --gpus all -p 8000:8000 \\"
echo "    -v \$(pwd)/checkpoints:/app/checkpoints:ro \\"
echo "    -e CHECKPOINT_PATH=/app/checkpoints/best_model.pth \\"
echo "    $FULL_IMAGE"
echo ""
echo "Or with docker-compose:"
echo "  docker-compose up -d"
