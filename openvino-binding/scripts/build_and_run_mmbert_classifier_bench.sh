#!/usr/bin/env bash
set -euo pipefail

# This script builds and runs the mmBERT classifier benchmark (OpenVINO vs Candle).
#
# Default behavior:
#   1) Build benchmark binary
#   2) Run benchmark binary
#
# Optional modes:
#   --build-only   Build binary only, do not run
#   --run-only     Run existing binary only

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/openvino-binding/bench"
BENCH_SRC="$BENCH_DIR/mmbert_classifier_bench.go"
BENCH_BIN="$BENCH_DIR/mmbert_classifier_bench"

# Defaults can be overridden by environment variables.
DEFAULT_CLASSIFIER_MODEL_DIR="models/mmbert-intent-classifier-merged"
DEFAULT_OPENVINO_TOKENIZERS_LIB="${OPENVINO_TOKENIZERS_LIB:-$(python3 -c "import openvino_tokenizers; import os; print(os.path.join(os.path.dirname(openvino_tokenizers.__file__), 'lib', 'libopenvino_tokenizers.so'))" 2>/dev/null || echo "")}"
DEFAULT_OPENVINO_LIB_DIR="${OPENVINO_LIB_DIR:-$(python3 -c "import openvino; import os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))" 2>/dev/null || echo "")}"

CLASSIFIER_MODEL_DIR="${CLASSIFIER_MODEL_DIR:-$DEFAULT_CLASSIFIER_MODEL_DIR}"
CLASSIFIER_OPENVINO_MODEL_PATH="${CLASSIFIER_OPENVINO_MODEL_PATH:-$CLASSIFIER_MODEL_DIR/openvino/openvino_model.xml}"
CLASSIFIER_CANDLE_MODEL_PATH="${CLASSIFIER_CANDLE_MODEL_PATH:-$CLASSIFIER_MODEL_DIR}"
OPENVINO_TOKENIZERS_LIB="${OPENVINO_TOKENIZERS_LIB:-$DEFAULT_OPENVINO_TOKENIZERS_LIB}"
OPENVINO_LIB_DIR="${OPENVINO_LIB_DIR:-$DEFAULT_OPENVINO_LIB_DIR}"
LD_LIBRARY_PATH="${ROOT_DIR}/candle-binding/target/release:${ROOT_DIR}/openvino-binding/build:${OPENVINO_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Correctness thresholds can be overridden from environment.
CLASSIFIER_MIN_MATCH_RATE="${CLASSIFIER_MIN_MATCH_RATE:-1.0}"
CLASSIFIER_MAX_CONFIDENCE_DELTA="${CLASSIFIER_MAX_CONFIDENCE_DELTA:-0.15}"
OPENVINO_CLASSIFIER_THREADS_PER_REQUEST="${OPENVINO_CLASSIFIER_THREADS_PER_REQUEST:-2}"
OPENVINO_CLASSIFIER_NUM_REQUESTS="${OPENVINO_CLASSIFIER_NUM_REQUESTS:-16}"
OPENVINO_CLASSIFIER_INFER_POOL_SIZE="${OPENVINO_CLASSIFIER_INFER_POOL_SIZE:-16}"

NUMA_CPUS="${NUMA_CPUS:-0-31}"
NUMA_NODE="${NUMA_NODE:-0}"

MODE="all"
if [[ "${1:-}" == "--build-only" ]]; then
  MODE="build"
elif [[ "${1:-}" == "--run-only" ]]; then
  MODE="run"
elif [[ "${1:-}" != "" ]]; then
  echo "Unsupported argument: $1"
  echo "Usage: $0 [--build-only|--run-only]"
  exit 1
fi

if [[ ! -f "$BENCH_SRC" ]]; then
  echo "Missing benchmark source: $BENCH_SRC"
  exit 1
fi

ensure_tokenizer_ir() {
  local model_dir="$1"
  local legacy_xml="$model_dir/openvino/tokenizer.xml"
  local legacy_bin="$model_dir/openvino/tokenizer.bin"
  local ov_xml="$model_dir/openvino/openvino_tokenizer.xml"
  local ov_bin="$model_dir/openvino/openvino_tokenizer.bin"

  # Accept either naming convention.
  if [[ (-f "$legacy_xml" && -f "$legacy_bin") || (-f "$ov_xml" && -f "$ov_bin") ]]; then
    return 0
  fi

  echo "Tokenizer IR not found for classifier model, generating tokenizer.xml/bin..."
  python -c "from transformers import AutoTokenizer; from openvino_tokenizers import convert_tokenizer; import openvino as ov; out='${legacy_xml}'; tok=AutoTokenizer.from_pretrained('${model_dir}', local_files_only=True); m=convert_tokenizer(tok, with_detokenizer=False); ov.save_model(m, out); print('saved', out)"
}

build_bench() {
  echo "[build] Building $BENCH_BIN ..."
  cd "$BENCH_DIR"
  # Auto-detect OpenVINO shared library name
  OV_SO="$(find "$OPENVINO_LIB_DIR" -maxdepth 1 -name 'libopenvino.so.*' -type f 2>/dev/null | head -1 | xargs -r basename)"
  if [[ -z "$OV_SO" ]]; then
    echo "Cannot find libopenvino.so.* in $OPENVINO_LIB_DIR"
    exit 1
  fi

  EXTRA_LIB_PATH=""
  EXTRA_LD_FLAGS=""
  if [[ -d "/usr/local/cuda/lib64" ]]; then
    EXTRA_LIB_PATH=":/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64"
    EXTRA_LD_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"
  fi

  env -u GOROOT \
    CGO_ENABLED=1 \
    LIBRARY_PATH="${OPENVINO_LIB_DIR}${EXTRA_LIB_PATH}" \
    CGO_LDFLAGS="-L$OPENVINO_LIB_DIR -l:$OV_SO $EXTRA_LD_FLAGS" \
    go build -v -o "$BENCH_BIN" "$BENCH_SRC"
  echo "[build] Done: $BENCH_BIN"
}

run_bench() {
  if [[ ! -x "$BENCH_BIN" ]]; then
    echo "Benchmark binary not found or not executable: $BENCH_BIN"
    echo "Run with --build-only first or default mode."
    exit 1
  fi

  if ! command -v numactl >/dev/null 2>&1; then
    echo "numactl is required for CPU/NUMA pinning but was not found in PATH"
    exit 1
  fi

  # Only check/generate tokenizer IR at run time.
  ensure_tokenizer_ir "$CLASSIFIER_MODEL_DIR"

  echo "[run] Running classifier benchmark..."
  echo "[run] CLASSIFIER_MODEL_DIR=$CLASSIFIER_MODEL_DIR"
  echo "[run] OPENVINO_MODEL_PATH=$CLASSIFIER_OPENVINO_MODEL_PATH"
  echo "[run] CANDLE_MODEL_PATH=$CLASSIFIER_CANDLE_MODEL_PATH"
  echo "[run] OPENVINO_TOKENIZERS_LIB=$OPENVINO_TOKENIZERS_LIB"
  echo "[run] CLASSIFIER_MIN_MATCH_RATE=$CLASSIFIER_MIN_MATCH_RATE"
  echo "[run] CLASSIFIER_MAX_CONFIDENCE_DELTA=$CLASSIFIER_MAX_CONFIDENCE_DELTA"
  echo "[run] NUMA_PIN=numactl -C ${NUMA_CPUS} --membind=${NUMA_NODE}"
  echo "[run] OPENVINO_CLASSIFIER_THREADS_PER_REQUEST=$OPENVINO_CLASSIFIER_THREADS_PER_REQUEST"
  echo "[run] OPENVINO_CLASSIFIER_NUM_REQUESTS=$OPENVINO_CLASSIFIER_NUM_REQUESTS"
  echo "[run] OPENVINO_CLASSIFIER_INFER_POOL_SIZE=$OPENVINO_CLASSIFIER_INFER_POOL_SIZE"

  cd "$BENCH_DIR"
  env -u GOROOT \
    OPENVINO_MODEL_PATH="$CLASSIFIER_OPENVINO_MODEL_PATH" \
    CANDLE_MODEL_PATH="$CLASSIFIER_CANDLE_MODEL_PATH" \
    OPENVINO_TOKENIZERS_LIB="$OPENVINO_TOKENIZERS_LIB" \
    CLASSIFIER_MIN_MATCH_RATE="$CLASSIFIER_MIN_MATCH_RATE" \
    CLASSIFIER_MAX_CONFIDENCE_DELTA="$CLASSIFIER_MAX_CONFIDENCE_DELTA" \
    OPENVINO_CLASSIFIER_THREADS_PER_REQUEST="$OPENVINO_CLASSIFIER_THREADS_PER_REQUEST" \
    OPENVINO_CLASSIFIER_NUM_REQUESTS="$OPENVINO_CLASSIFIER_NUM_REQUESTS" \
    OPENVINO_CLASSIFIER_INFER_POOL_SIZE="$OPENVINO_CLASSIFIER_INFER_POOL_SIZE" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    numactl -C "${NUMA_CPUS}" --membind="${NUMA_NODE}" "$BENCH_BIN"
}

if [[ "$MODE" == "build" ]]; then
  build_bench
elif [[ "$MODE" == "run" ]]; then
  run_bench
else
  build_bench
  run_bench
fi
