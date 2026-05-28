#!/usr/bin/env bash
set -euo pipefail

# This script builds and runs the mmBERT embedding benchmark (OpenVINO vs Candle).
# Default behavior:
#   1) Build benchmark binary
#   2) Run benchmark binary
#
# Optional modes:
#   --build-only                 Build binary only, do not run
#   --run-only                   Run existing binary only
#   --stage-timing               Enable 3-stage timing diagnostics
#   --stage-timing-samples <N>   Number of samples for stage timing probe (default: 10)
#   --length-profile <mode>      mixed|fixed-32|fixed-128|fixed-512|fixed-1024|fixed-2048 (default: mixed)

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/openvino-binding/bench"
BENCH_SRC="$BENCH_DIR/mmbert_embedding_bench.go"
BENCH_BIN="$BENCH_DIR/mmbert_embedding_bench"

# Defaults can be overridden by environment variables.
DEFAULT_MMBERT_MODEL_PATH="models/mmbert-embed-32k-2d-matryoshka"
DEFAULT_OPENVINO_TOKENIZERS_LIB="${OPENVINO_TOKENIZERS_LIB:-$(python3 -c "import openvino_tokenizers; import os; print(os.path.join(os.path.dirname(openvino_tokenizers.__file__), 'lib', 'libopenvino_tokenizers.so'))" 2>/dev/null || echo "")}"
DEFAULT_OPENVINO_LIB_DIR="${OPENVINO_LIB_DIR:-$(python3 -c "import openvino; import os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))" 2>/dev/null || echo "")}"

MMBERT_MODEL_PATH="${MMBERT_MODEL_PATH:-$DEFAULT_MMBERT_MODEL_PATH}"
OPENVINO_MODEL_PATH="${OPENVINO_MODEL_PATH:-$MMBERT_MODEL_PATH/openvino/openvino_model.xml}"
CANDLE_MODEL_PATH="${CANDLE_MODEL_PATH:-$MMBERT_MODEL_PATH}"
OPENVINO_TOKENIZERS_LIB="${OPENVINO_TOKENIZERS_LIB:-$DEFAULT_OPENVINO_TOKENIZERS_LIB}"
OPENVINO_LIB_DIR="${OPENVINO_LIB_DIR:-$DEFAULT_OPENVINO_LIB_DIR}"
LD_LIBRARY_PATH="${ROOT_DIR}/candle-binding/target/release:${ROOT_DIR}/openvino-binding/build:${OPENVINO_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

NUMA_CPUS="${NUMA_CPUS:-0-31}"
NUMA_NODE="${NUMA_NODE:-0}"

MODE="all"
STAGE_TIMING="${EMBEDDING_STAGE_TIMING:-0}"
STAGE_TIMING_SAMPLES="${EMBEDDING_STAGE_TIMING_SAMPLES:-10}"
LENGTH_PROFILE="${EMBEDDING_LENGTH_PROFILE:-mixed}"
OV_MAX_LENGTH="${OV_MAX_LENGTH:-512}"
OPENVINO_EMBEDDING_THREADS_PER_REQUEST="${OPENVINO_EMBEDDING_THREADS_PER_REQUEST:-2}"
OPENVINO_EMBEDDING_NUM_REQUESTS="${OPENVINO_EMBEDDING_NUM_REQUESTS:-16}"
OPENVINO_EMBEDDING_INFER_POOL_SIZE="${OPENVINO_EMBEDDING_INFER_POOL_SIZE:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)
      MODE="build"
      shift
      ;;
    --run-only)
      MODE="run"
      shift
      ;;
    --stage-timing)
      STAGE_TIMING="1"
      shift
      ;;
    --stage-timing-samples)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --stage-timing-samples"
        exit 1
      fi
      STAGE_TIMING_SAMPLES="$2"
      shift 2
      ;;
    --length-profile)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --length-profile"
        exit 1
      fi
      LENGTH_PROFILE="$2"
      shift 2
      ;;
    *)
      echo "Unsupported argument: $1"
      echo "Usage: $0 [--build-only|--run-only] [--stage-timing] [--stage-timing-samples N] [--length-profile MODE]"
      exit 1
      ;;
  esac
done

# Ensure benchmark source exists.
if [[ ! -f "$BENCH_SRC" ]]; then
  echo "Missing benchmark source: $BENCH_SRC"
  exit 1
fi

# Ensure OpenVINO tokenizer IR exists (tokenizer.xml/bin or openvino_tokenizer.xml/bin).
TOKENIZER_XML="$MMBERT_MODEL_PATH/openvino/tokenizer.xml"
TOKENIZER_BIN="$MMBERT_MODEL_PATH/openvino/tokenizer.bin"
OV_TOKENIZER_XML="$MMBERT_MODEL_PATH/openvino/openvino_tokenizer.xml"
OV_TOKENIZER_BIN="$MMBERT_MODEL_PATH/openvino/openvino_tokenizer.bin"
if [[ ! (( -f "$TOKENIZER_XML" && -f "$TOKENIZER_BIN" ) || ( -f "$OV_TOKENIZER_XML" && -f "$OV_TOKENIZER_BIN" )) ]]; then
  echo "Tokenizer IR not found, generating tokenizer.xml/bin..."
  python -c "from transformers import AutoTokenizer; from openvino_tokenizers import convert_tokenizer; import openvino as ov; out='${TOKENIZER_XML}'; tok=AutoTokenizer.from_pretrained('${MMBERT_MODEL_PATH}', local_files_only=True); m=convert_tokenizer(tok, with_detokenizer=False); ov.save_model(m, out); print('saved', out)"
fi

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

  echo "[run] Running benchmark..."
  echo "[run] MMBERT_MODEL_PATH=$MMBERT_MODEL_PATH"
  echo "[run] OPENVINO_MODEL_PATH=$OPENVINO_MODEL_PATH"
  echo "[run] CANDLE_MODEL_PATH=$CANDLE_MODEL_PATH"
  echo "[run] OPENVINO_TOKENIZERS_LIB=$OPENVINO_TOKENIZERS_LIB"
  echo "[run] NUMA_PIN=numactl -C ${NUMA_CPUS} --membind=${NUMA_NODE}"
  echo "[run] EMBEDDING_STAGE_TIMING=$STAGE_TIMING"
  echo "[run] EMBEDDING_STAGE_TIMING_SAMPLES=$STAGE_TIMING_SAMPLES"
  echo "[run] EMBEDDING_LENGTH_PROFILE=$LENGTH_PROFILE"
  echo "[run] OV_MAX_LENGTH=$OV_MAX_LENGTH"
  echo "[run] OPENVINO_EMBEDDING_THREADS_PER_REQUEST=$OPENVINO_EMBEDDING_THREADS_PER_REQUEST"
  echo "[run] OPENVINO_EMBEDDING_NUM_REQUESTS=$OPENVINO_EMBEDDING_NUM_REQUESTS"
  echo "[run] OPENVINO_EMBEDDING_INFER_POOL_SIZE=$OPENVINO_EMBEDDING_INFER_POOL_SIZE"

  cd "$BENCH_DIR"
  env -u GOROOT \
    OPENVINO_MODEL_PATH="$OPENVINO_MODEL_PATH" \
    CANDLE_MODEL_PATH="$CANDLE_MODEL_PATH" \
    OPENVINO_TOKENIZERS_LIB="$OPENVINO_TOKENIZERS_LIB" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    EMBEDDING_STAGE_TIMING="$STAGE_TIMING" \
    EMBEDDING_STAGE_TIMING_SAMPLES="$STAGE_TIMING_SAMPLES" \
    EMBEDDING_LENGTH_PROFILE="$LENGTH_PROFILE" \
    OV_MAX_LENGTH="$OV_MAX_LENGTH" \
    OPENVINO_EMBEDDING_THREADS_PER_REQUEST="$OPENVINO_EMBEDDING_THREADS_PER_REQUEST" \
    OPENVINO_EMBEDDING_NUM_REQUESTS="$OPENVINO_EMBEDDING_NUM_REQUESTS" \
    OPENVINO_EMBEDDING_INFER_POOL_SIZE="$OPENVINO_EMBEDDING_INFER_POOL_SIZE" \
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
