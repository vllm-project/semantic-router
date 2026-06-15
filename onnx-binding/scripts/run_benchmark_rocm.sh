#!/bin/bash
# Run ort-binding benchmark with ROCm GPU inside Docker
# Must run build_rocm.sh first

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="rocm/onnxruntime:rocm7.0_ub22.04_ort1.22_torch2.8.0"
ORT_LIB="/opt/venv/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.22.1"
ORT_LIB_DIR="/opt/venv/lib/python3.10/site-packages/onnxruntime/capi"

# Check if binary exists
if [ ! -f "$PROJECT_DIR/target/release/examples/benchmark_mmbert_latency" ]; then
    echo "Binary not found. Run build_rocm.sh first."
    exit 1
fi

echo "============================================"
echo "Running mmBERT Latency Benchmark (ROCm GPU)"
echo "============================================"
echo ""

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$PROJECT_DIR":/workspace \
    -v /data:/data \
    -w /workspace \
    -e ORT_DYLIB_PATH="$ORT_LIB" \
    -e LD_LIBRARY_PATH="$ORT_LIB_DIR:/opt/rocm/lib" \
    -e ORT_LOG=warning \
    "$IMAGE" \
    ./target/release/examples/benchmark_mmbert_latency
