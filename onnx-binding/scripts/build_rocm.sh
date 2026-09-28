#!/bin/bash
# Build and run ort-binding with ROCm GPU support in Docker
# This script handles the ORT version compatibility automatically

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Docker image with matching ROCm + ORT versions
IMAGE="rocm/onnxruntime:rocm7.0_ub22.04_ort1.22_torch2.8.0"

# ORT library path inside container
ORT_LIB="/opt/venv/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.1"
ORT_LIB_DIR="/opt/venv/lib/python3.10/site-packages/onnxruntime/capi"

echo "============================================"
echo "Building ort-binding with ROCm GPU support"
echo "============================================"
echo "Image: $IMAGE"
echo ""

# Run build inside container
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$PROJECT_DIR":/workspace \
    -v "$HOME/.cargo/registry":/root/.cargo/registry \
    -v "$HOME/.cargo/git":/root/.cargo/git \
    -v /data:/data \
    -w /workspace \
    -e ORT_DYLIB_PATH="$ORT_LIB" \
    -e LD_LIBRARY_PATH="$ORT_LIB_DIR:/opt/rocm/lib" \
    "$IMAGE" \
    bash -c '
        # Install Rust if not available
        if ! command -v rustc &> /dev/null; then
            echo "Installing Rust..."
            curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source $HOME/.cargo/env
        fi
        export PATH="$HOME/.cargo/bin:$PATH"
        
        echo "Rust version: $(rustc --version)"
        echo "ORT library: $ORT_DYLIB_PATH"
        echo ""
        
        # Cargo.lock pins ort/ort-sys rc.10, whose native API is 22.
        # Keep the matching ORT 1.22 runtime; never patch registry sources or
        # disable the native ABI version check to force compatibility.
        cargo fetch --locked

        # Build
        echo ""
        echo "Building with rocm-dynamic feature..."
        cargo build --release --features rocm-dynamic --examples
        
        echo ""
        echo "Build complete!"
    '

echo ""
echo "============================================"
echo "Build successful!"
echo "============================================"
