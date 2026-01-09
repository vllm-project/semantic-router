#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"

echo "Starting Semantic Router Multi-Model Demo"
echo ""

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Error: Docker required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: Python 3 required"; exit 1; }

# Download models
echo "Downloading models (~24GB if missing)..."
mkdir -p "$MODELS_DIR"

download_model() {
    local repo="$1"
    local local_dir="$2"
    [ -d "$local_dir" ] && return
    echo "  Downloading $(basename "$local_dir")..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='$local_dir', local_dir_use_symlinks=False)"
}

download_model "Qwen/Qwen2.5-3B-Instruct" "$MODELS_DIR/Qwen/Qwen2.5-3B-Instruct"
download_model "nvidia/NVIDIA-Nemotron-Nano-9B-v2" "$MODELS_DIR/NVIDIA-Nemotron-Nano-9B-v2"

# Start vLLM servers
echo "Starting vLLM servers..."
cd "$SCRIPT_DIR"
export MODELS_DIR="$MODELS_DIR"
docker compose -f docker-compose-models.yml up -d

# Wait for vLLM
echo "Waiting for models (1-3 min)..."
for i in {1..60}; do
    curl -sf http://localhost:8000/health && break
    sleep 3
done
for i in {1..120}; do
    curl -sf http://localhost:8002/health && break
    sleep 3
done

# Start router
echo "Starting semantic router..."
vllm-sr serve --config "$SCRIPT_DIR/config.yaml" >/dev/null 2>&1 &

# Wait for router
for i in {1..30}; do
    curl -sf http://localhost:8888/health && break
    sleep 2
done

echo ""
echo "Demo ready"
echo ""
echo "Test: python3 run_demo.py"
echo "Stop: vllm-sr stop && docker compose -f docker-compose-models.yml down"
echo ""
