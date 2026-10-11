#!/usr/bin/env bash
# Run the current benchmark harness against a real base revision. Both sides
# serve the same pinned catalog models through the model runtime, each side
# with its own runtime source. No baseline numbers are invented or copied from
# a different implementation.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_REF="${1:?usage: compare-model-baseline.sh BASE_REF OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: compare-model-baseline.sh BASE_REF OUTPUT_DIR}"
for model_env in VLLM_SR_DOMAIN_MODEL VLLM_SR_PII_MODEL VLLM_SR_JAILBREAK_MODEL VLLM_SR_EMBEDDING_MODEL; do
  model_path="${!model_env:-}"
  if [[ -n "$model_path" && "$model_path" != /* ]]; then
    export "$model_env=$ROOT_DIR/$model_path"
  fi
done
BASE_COMMIT="$(git -C "$ROOT_DIR" rev-parse --verify "${BASE_REF}^{commit}")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
BASE_TEMP="$(mktemp -d)"
BASE_TREE="$BASE_TEMP/source"
cleanup() {
  git -C "$ROOT_DIR" worktree remove --force "$BASE_TREE" >/dev/null 2>&1 || true
  rm -rf "$BASE_TEMP"
}
trap cleanup EXIT

git -C "$ROOT_DIR" worktree add --detach "$BASE_TREE" "$BASE_COMMIT"

if [[ ! -d "$BASE_TREE/src/semantic-router/pkg/modelruntime/serving" ]]; then
  # The base revision runs models in the removed native bindings, so no common
  # harness can measure both sides. Record the reset instead of a comparison,
  # citing the records that measured the bindings against the runtime.
  records=()
  shopt -s nullglob
  for record in "$ROOT_DIR"/src/model-runtime/docs/records/{router-,vela1-,embed-,decision1-,stores-algorithms}*; do
    records+=("${record#"$ROOT_DIR"/}")
  done
  shopt -u nullglob
  if ((${#records[@]} == 0)); then
    echo "A model baseline reset must cite the legacy-versus-runtime records; none found." >&2
    exit 1
  fi
  python3 - "$BASE_COMMIT" "$OUTPUT_DIR/model-baseline.json" "${records[@]}" <<'PYCODE'
import json
import sys
json.dump(
    {
        "git_commit": sys.argv[1],
        "model_baseline_reset": "base revision predates the model runtime; no common model harness",
        "legacy_comparison_records": sorted(sys.argv[3:]),
        "benchmarks": {},
    },
    open(sys.argv[2], "w"),
    indent=2,
)
PYCODE
  echo "Model baseline reset: $BASE_COMMIT predates the model runtime."
  exit 0
fi

# The measurement program is held constant; only the implementation changes.
rm -rf "$BASE_TREE/perf"
cp -R "$ROOT_DIR/perf" "$BASE_TREE/perf"

# The base side runs its own runtime source on the installed dependencies.
# Before #4618 the runtime was vllm_sr_runtime: it read VLLM_SR_RUNTIME_* and
# found its plugins under its own entry-point groups, which the installed
# vllm-srun doesn't declare. Such a base gets its package, with that metadata,
# and the names it reads.
if [[ -d "$BASE_TREE/src/model-runtime/vllm_srun" ]]; then
  export PYTHONPATH="$BASE_TREE/src/model-runtime${PYTHONPATH:+:$PYTHONPATH}"
  export VLLM_SRUN_COMMAND="${PERF_RUNTIME_PYTHON:-python3} -m vllm_srun"
else
  "${PERF_RUNTIME_PYTHON:-python3}" -m pip install --quiet --no-deps \
    --target "$BASE_TEMP/runtime" "$BASE_TREE/src/model-runtime"
  export PYTHONPATH="$BASE_TEMP/runtime${PYTHONPATH:+:$PYTHONPATH}"
  export VLLM_SR_RUNTIME_COMMAND="${PERF_RUNTIME_PYTHON:-python3} -m vllm_sr_runtime"
  if [[ -n "${VLLM_SRUN_CACHE_DIR:-}" ]]; then
    export VLLM_SR_RUNTIME_CACHE_DIR="$VLLM_SRUN_CACHE_DIR"
  fi
fi
(
  cd "$BASE_TREE/perf"
  export GIT_WORK_TREE="$BASE_TREE"
  if ! go test -run '^$' -bench '^(BenchmarkClassify|BenchmarkCache)' \
    -benchmem -benchtime="${PERF_MODEL_BENCHTIME:-3s}" -timeout=30m ./benchmarks/... \
    2>&1 | tee "$OUTPUT_DIR/model-baseline-output.txt"; then
    echo "Base revision $BASE_COMMIT cannot run the current model benchmark contract; inspect compile/runtime output and choose an explicitly compatible base." >&2
    exit 1
  fi
  go run ./cmd/perftest --parse-bench="$OUTPUT_DIR/model-baseline-output.txt" \
    --output="$OUTPUT_DIR/model-baseline.json"
)
