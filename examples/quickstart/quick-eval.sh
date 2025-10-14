#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="${SCRIPT_DIR}/results"
DEFAULT_DATASET="mmlu"
DEFAULT_SAMPLES=${QUICKSTART_SAMPLES:-5}
DEFAULT_MODE="${QUICKSTART_MODE:-router}"
DEFAULT_ROUTER_MODELS="${QUICKSTART_ROUTER_MODELS:-auto}"
DEFAULT_VLLM_MODELS="${QUICKSTART_VLLM_MODELS:-openai/gpt-oss-20b}"
DEFAULT_ROUTER_ENDPOINT="${ROUTER_ENDPOINT:-http://127.0.0.1:8801/v1}"
DEFAULT_VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}"
DEFAULT_SEED=${QUICKSTART_SEED:-42}
DEFAULT_TEMPERATURE="${QUICKSTART_TEMPERATURE:-0.01}"
REQUIRED_COMMANDS=()
REQUIRED_PY_MODULES=(numpy pandas datasets openai)

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo '[ERROR] No python interpreter found in PATH or VIRTUAL_ENV' >&2
    exit 1
  fi
fi

usage() {
  cat <<'USAGE'
Usage: quick-eval.sh [OPTIONS]

Options:
  --dataset NAME          Dataset identifier (default: mmlu)
  --samples N             Samples per category (default: 5 or $QUICKSTART_SAMPLES)
  --mode MODE             router|vllm|both (default: router or $QUICKSTART_MODE)
  --router-endpoint URL   Router endpoint (default: env ROUTER_ENDPOINT or http://127.0.0.1:8801/v1)
  --vllm-endpoint URL     vLLM endpoint (default: env VLLM_ENDPOINT or http://127.0.0.1:8000/v1)
  --router-models LIST    Space-separated router models (default: auto)
  --vllm-models LIST      Space-separated vLLM models (default: openai/gpt-oss-20b)
  --output-dir DIR        Directory to store run artifacts (default: examples/quickstart/results/<timestamp>)
  --seed N                Random seed (default: 42 or $QUICKSTART_SEED)
  --temperature FLOAT     Sampling temperature (default: 0.01 or $QUICKSTART_TEMPERATURE)
  --help                  Show this help message

Environment overrides:
  QUICKSTART_SAMPLES, QUICKSTART_MODE, QUICKSTART_ROUTER_MODELS,
  QUICKSTART_VLLM_MODELS, QUICKSTART_SEED, QUICKSTART_TEMPERATURE,
  ROUTER_ENDPOINT, VLLM_ENDPOINT.

The script launches the benchmark module with quickstart-friendly defaults and
emits both CSV and Markdown summaries for the generated results.
USAGE
}

log() {
  local level="$1"
  shift
  printf '[%s] %s\n' "$level" "$*"
}

die() {
  log "ERROR" "$*"
  exit 1
}

parse_args() {
  DATASET="$DEFAULT_DATASET"
  SAMPLES="$DEFAULT_SAMPLES"
  MODE="$DEFAULT_MODE"
  ROUTER_ENDPOINT="$DEFAULT_ROUTER_ENDPOINT"
  VLLM_ENDPOINT="$DEFAULT_VLLM_ENDPOINT"
  ROUTER_MODELS=("$DEFAULT_ROUTER_MODELS")
  VLLM_MODELS=("$DEFAULT_VLLM_MODELS")
  OUTPUT_DIR=""
  SEED="$DEFAULT_SEED"
  TEMPERATURE="$DEFAULT_TEMPERATURE"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)
        DATASET="$2"
        shift 2
        ;;
      --samples)
        SAMPLES="$2"
        shift 2
        ;;
      --mode)
        MODE="$2"
        shift 2
        ;;
      --router-endpoint)
        ROUTER_ENDPOINT="$2"
        shift 2
        ;;
      --vllm-endpoint)
        VLLM_ENDPOINT="$2"
        shift 2
        ;;
      --router-models)
        shift
        ROUTER_MODELS=()
        while [[ $# -gt 0 ]] && [[ $1 != --* ]]; do
          ROUTER_MODELS+=("$1")
          shift
        done
        ;;
      --vllm-models)
        shift
        VLLM_MODELS=()
        while [[ $# -gt 0 ]] && [[ $1 != --* ]]; do
          VLLM_MODELS+=("$1")
          shift
        done
        ;;
      --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --seed)
        SEED="$2"
        shift 2
        ;;
      --temperature)
        TEMPERATURE="$2"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done

  case "$MODE" in
    router|vllm|both) ;;
    *) die "Invalid --mode '$MODE' (expected router|vllm|both)" ;;
  esac

  if [[ -z "$OUTPUT_DIR" ]]; then
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${RESULTS_ROOT}/${timestamp}"
  fi
}

require_commands() {
  local missing=()
  for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing+=("$cmd")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    die "Missing required commands: ${missing[*]}"
  fi
}

check_python_modules() {
  if ! "$PYTHON_BIN" - "${REQUIRED_PY_MODULES[@]}" <<'PY'; then
import importlib
import sys
missing = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
PY
    die "Missing Python modules. Run 'pip install -r bench/requirements.txt'"
  fi
}

prepare_dirs() {
  mkdir -p "$RESULTS_ROOT"
  mkdir -p "$OUTPUT_DIR"
}

run_benchmark() {
  local raw_dir="${OUTPUT_DIR}/raw"
  mkdir -p "$raw_dir"

  local cmd=("$PYTHON_BIN" -m vllm_semantic_router_bench.router_reason_bench_multi_dataset
    --dataset "$DATASET"
    --samples-per-category "$SAMPLES"
    --output-dir "$raw_dir"
    --seed "$SEED"
    --temperature "$TEMPERATURE"
    --router-endpoint "$ROUTER_ENDPOINT"
  )

  if [[ "$MODE" == "router" || "$MODE" == "both" ]]; then
    cmd+=(--run-router --router-models "${ROUTER_MODELS[@]}")
  fi

  if [[ "$MODE" == "vllm" || "$MODE" == "both" ]]; then
    cmd+=(--run-vllm --vllm-endpoint "$VLLM_ENDPOINT" --vllm-models "${VLLM_MODELS[@]}" --vllm-exec-modes NR)
  fi

  log "INFO" "Running benchmark: ${cmd[*]}"
  (
    cd "$ROOT_DIR"
    PYTHONPATH="$ROOT_DIR/bench${PYTHONPATH:+:$PYTHONPATH}" \
    ROUTER_ENDPOINT="$ROUTER_ENDPOINT" \
    VLLM_ENDPOINT="$VLLM_ENDPOINT" \
      "${cmd[@]}"
  ) || die "Benchmark run failed"

  RAW_DIR="$raw_dir"
}

collect_summaries() {
  mapfile -t SUMMARY_FILES < <(find "$RAW_DIR" -type f -name summary.json -print | sort)
  if [[ ${#SUMMARY_FILES[@]} -eq 0 ]]; then
    die "No summary.json files produced under $RAW_DIR"
  fi

  SUMMARY_CSV="$OUTPUT_DIR/quickstart-summary.csv"
  REPORT_MD="$OUTPUT_DIR/quickstart-report.md"

  "$PYTHON_BIN" - "$SUMMARY_CSV" "$REPORT_MD" "${SUMMARY_FILES[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

summary_csv = Path(sys.argv[1])
report_md = Path(sys.argv[2])
summary_paths = [Path(p) for p in sys.argv[3:]]

rows = []
for path in summary_paths:
    data = json.loads(path.read_text())
    rows.append({
        "dataset": data.get("dataset"),
        "model": data.get("model"),
        "overall_accuracy": data.get("overall_accuracy"),
        "avg_response_time": data.get("avg_response_time"),
        "avg_total_tokens": data.get("avg_total_tokens"),
        "total_questions": data.get("total_questions"),
        "successful_queries": data.get("successful_queries"),
        "failed_queries": data.get("failed_queries"),
        "source": str(path.relative_to(summary_csv.parent))
    })

rows.sort(key=lambda r: (r["dataset"] or "", r["model"] or ""))

csv_lines = ["dataset,model,overall_accuracy,avg_response_time,avg_total_tokens,total_questions,successful_queries,failed_queries,source"]
for row in rows:
    csv_lines.append(
        f"{row['dataset']},{row['model']},{row['overall_accuracy']},{row['avg_response_time']},{row['avg_total_tokens']},{row['total_questions']},{row['successful_queries']},{row['failed_queries']},{row['source']}"
    )
summary_csv.write_text("\n".join(csv_lines) + "\n")

lines = ["# Quickstart Evaluation Report", "", f"Generated: {datetime.utcnow().isoformat()}Z", ""]
lines.append("| Dataset | Model | Accuracy | Avg Latency (s) | Avg Tokens | Samples |")
lines.append("| --- | --- | --- | --- | --- | --- |")
for row in rows:
    accuracy = f"{row['overall_accuracy']:.3f}" if isinstance(row['overall_accuracy'], (int, float)) else "N/A"
    latency = f"{row['avg_response_time']:.2f}" if isinstance(row['avg_response_time'], (int, float)) else "N/A"
    tokens = f"{row['avg_total_tokens']:.1f}" if isinstance(row['avg_total_tokens'], (int, float)) else "N/A"
    total = row['total_questions'] if row['total_questions'] is not None else 0
    success = row['successful_queries'] if row['successful_queries'] is not None else 0
    lines.append(f"| {row['dataset']} | {row['model']} | {accuracy} | {latency} | {tokens} | {success}/{total} |")

lines.append("")
lines.append("## Source Artifacts")
lines.append("")
for row in rows:
    lines.append(f"- `{row['source']}`")

report_md.write_text("\n".join(lines) + "\n")
PY

  log "INFO" "Summary CSV: $SUMMARY_CSV"
  log "INFO" "Report Markdown: $REPORT_MD"
}

main() {
  parse_args "$@"
  require_commands
  log "INFO" "Using python interpreter: ${PYTHON_BIN}"
  check_python_modules
  prepare_dirs
  run_benchmark
  collect_summaries
}

main "$@"
