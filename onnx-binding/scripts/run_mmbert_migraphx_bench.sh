#!/usr/bin/env bash
# Build and run the mmBERT ONNX Runtime provider benchmark for AMD MIGraphX.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="${MMBERT_MODEL_DIR:-/models/mmbert-embed-32k-2d-matryoshka/onnx}"
FEATURES="${FEATURES:-migraphx-dynamic}"
EXAMPLE="${EXAMPLE:-benchmark_mmbert_ort_providers}"
DEFAULT_PROVIDERS="${PROVIDERS:-cpu,migraphx,rocm}"
DEFAULT_LAYERS="${LAYERS:-6,22}"
DEFAULT_SEQ_LENS="${SEQ_LENS:-1,32,128}"
DEFAULT_BATCH_SIZES="${BATCH_SIZES:-1}"
DEFAULT_WARMUP="${WARMUP:-3}"
DEFAULT_ITERS="${ITERS:-20}"
DEFAULT_SUMMARY_MD="${SUMMARY_MD:-}"

find_ort_lib() {
    if [[ -n "${ORT_DYLIB_PATH:-}" && -f "${ORT_DYLIB_PATH}" ]]; then
        printf '%s\n' "${ORT_DYLIB_PATH}"
        return 0
    fi

    local candidates=(
        /tmp/ort_migx_71/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.1
        /usr/local/lib/python3.12/dist-packages/onnxruntime/capi/libonnxruntime.so.1.23.1
        /usr/local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.1
        /opt/venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.1
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

has_flag() {
    local flag="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$flag" ]]; then
            return 0
        fi
    done
    return 1
}

if ORT_LIB="$(find_ort_lib)"; then
    export ORT_DYLIB_PATH="$ORT_LIB"
    ORT_LIB_DIR="$(dirname "$ORT_LIB")"
    export LD_LIBRARY_PATH="/opt/rocm/lib:${ORT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    ORT_SITE_PACKAGES="${ORT_LIB_DIR%/onnxruntime/capi}"
    if [[ -d "$ORT_SITE_PACKAGES" ]]; then
        export PYTHONPATH="${ORT_SITE_PACKAGES}:${PYTHONPATH:-}"
    fi
else
    echo "ERROR: could not find AMD ONNX Runtime MIGraphX libonnxruntime.so.1.23.1" >&2
    echo "Set ORT_DYLIB_PATH or install the AMD onnxruntime_migraphx 1.23.1 wheel." >&2
    exit 1
fi

echo "============================================================"
echo "mmBERT ORT/MIGraphX provider benchmark"
echo "============================================================"
echo "project:        $PROJECT_DIR"
echo "model dir:      $MODEL_DIR"
echo "features:       $FEATURES"
echo "ORT_DYLIB_PATH: $ORT_DYLIB_PATH"
echo "LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
echo

if command -v migraphx-driver >/dev/null 2>&1; then
    migraphx-driver --version || true
else
    echo "WARN: migraphx-driver not found on PATH"
fi

if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' || true
try:
    import onnxruntime as ort
    print("python_onnxruntime_version:", ort.__version__)
    print("python_onnxruntime_providers:", ",".join(ort.get_available_providers()))
except Exception as exc:
    print("WARN: python onnxruntime provider inventory failed:", exc)
PY
fi

args=("$@")
if ! has_flag "--model-dir" "${args[@]}"; then
    args+=("--model-dir" "$MODEL_DIR")
fi
if ! has_flag "--providers" "${args[@]}"; then
    args+=("--providers" "$DEFAULT_PROVIDERS")
fi
if ! has_flag "--layers" "${args[@]}"; then
    args+=("--layers" "$DEFAULT_LAYERS")
fi
if ! has_flag "--seq-lens" "${args[@]}"; then
    args+=("--seq-lens" "$DEFAULT_SEQ_LENS")
fi
if ! has_flag "--batch-sizes" "${args[@]}"; then
    args+=("--batch-sizes" "$DEFAULT_BATCH_SIZES")
fi
if ! has_flag "--warmup" "${args[@]}"; then
    args+=("--warmup" "$DEFAULT_WARMUP")
fi
if ! has_flag "--iters" "${args[@]}"; then
    args+=("--iters" "$DEFAULT_ITERS")
fi
if [[ -n "$DEFAULT_SUMMARY_MD" ]] && ! has_flag "--summary-md" "${args[@]}"; then
    args+=("--summary-md" "$DEFAULT_SUMMARY_MD")
fi

cd "$PROJECT_DIR"

if command -v cargo >/dev/null 2>&1; then
    cargo run --release --features "$FEATURES" --example "$EXAMPLE" -- "${args[@]}"
else
    binary="$PROJECT_DIR/target/release/examples/$EXAMPLE"
    if [[ ! -x "$binary" ]]; then
        echo "ERROR: cargo is unavailable and benchmark binary was not found: $binary" >&2
        exit 1
    fi
    "$binary" "${args[@]}"
fi
