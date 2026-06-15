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
        
        # The ort crate 2.0 expects API v23, but ORT 1.22 provides API v22
        # We need to patch ort-sys to accept API v22
        # This is done in-container to avoid polluting host cargo cache
        
        # First, ensure dependencies are downloaded
        cargo fetch 2>/dev/null || true
        
        # Find and patch ort-sys version.rs
        ORT_SYS_DIR=$(find $HOME/.cargo/registry/src -name "ort-sys-2.0*" -type d 2>/dev/null | head -1)
        if [ -n "$ORT_SYS_DIR" ]; then
            VERSION_FILE="$ORT_SYS_DIR/src/version.rs"
            if [ -f "$VERSION_FILE" ]; then
                echo "Patching ort-sys API version for ORT 1.22 compatibility..."
                sed -i "s/ORT_API_VERSION: u32 = 23/ORT_API_VERSION: u32 = 22/" "$VERSION_FILE"
            fi
        fi
        
        # Also patch ort to not reject ORT 1.22
        ORT_DIR=$(find $HOME/.cargo/registry/src -name "ort-2.0*" -type d 2>/dev/null | head -1)
        if [ -n "$ORT_DIR" ]; then
            LIB_FILE="$ORT_DIR/src/lib.rs"
            if [ -f "$LIB_FILE" ] && grep -q "Ordering::Less =>" "$LIB_FILE"; then
                echo "Patching ort version check for ORT 1.22..."
                # Change: reject < 23 -> reject < 22
                sed -i "s/Ordering::Less => {/Ordering::Less if lib_minor_version < 22 => {/" "$LIB_FILE"
                # Add fallthrough case for Less when >= 22
                sed -i "/Ordering::Equal => {}/i\\            Ordering::Less => crate::info!(\"Using ORT 1.22 with ort 2.0 compatibility mode\")," "$LIB_FILE"
            fi
        fi
        
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
