#!/bin/bash
# Build onnx-binding with AMD MIGraphX-first provider support in Docker.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Docker image with matching ROCm + MIGraphX package versions.
IMAGE="rocm/dev-ubuntu-24.04:7.1"
ORT_WHEEL_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/onnxruntime_migraphx-1.23.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

# ORT library path inside container
ORT_LIB="/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/libonnxruntime.so.1.23.1"
ORT_LIB_DIR="/usr/local/lib/python3.12/dist-packages/onnxruntime/capi"

echo "============================================"
echo "Building onnx-binding with AMD MIGraphX-first support"
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
    -e ORT_WHEEL_URL="$ORT_WHEEL_URL" \
    -e ORT_DYLIB_PATH="$ORT_LIB" \
    -e LD_LIBRARY_PATH="$ORT_LIB_DIR:/opt/rocm/lib" \
    "$IMAGE" \
    bash -c '
        apt-get update
        apt-get install -y --no-install-recommends \
            ca-certificates curl hipblas hipfft migraphx miopen-hip python3-pip rccl
        python3 -m pip install --no-cache-dir --break-system-packages "$ORT_WHEEL_URL"
        ln -sf "$ORT_LIB" "$ORT_LIB_DIR/libonnxruntime.so"

        migraphx-driver --version || true
        python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

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

        cargo fetch 2>/dev/null || true

        # Build
        echo ""
        echo "Building with migraphx-dynamic feature..."
        cargo build --release --features migraphx-dynamic --examples
        
        echo ""
        echo "Build complete!"
    '

echo ""
echo "============================================"
echo "Build successful!"
echo "============================================"
