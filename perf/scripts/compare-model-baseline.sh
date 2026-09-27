#!/usr/bin/env bash
# Run the current benchmark harness against a real base revision, using exactly
# the current run's pinned artifacts. No baseline numbers are invented or copied
# from a different model family.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_REF="${1:?usage: compare-model-baseline.sh BASE_REF OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: compare-model-baseline.sh BASE_REF OUTPUT_DIR}"
: "${VLLM_SR_MODEL_MANIFEST:?set the absolute manifest from make download-models-perf}"
if [[ "$VLLM_SR_MODEL_MANIFEST" != /* ]]; then
  echo "VLLM_SR_MODEL_MANIFEST must be absolute so both revisions read the same artifacts" >&2
  exit 2
fi
# The downloader emits absolute paths. Reject ambiguous manifests before creating
# the base worktree, and anchor explicit relative overrides to the current tree.
python3 - "$VLLM_SR_MODEL_MANIFEST" <<'PYCODE'
import json
import pathlib
import sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("provider") != "candle":
    raise SystemExit("model baseline requires a Candle manifest")
for model in manifest["models"]:
    if not pathlib.Path(model["path"]).is_absolute():
        raise SystemExit("model baseline requires absolute artifact paths")
PYCODE
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
# The measurement program is held constant; only the implementation changes.
rm -rf "$BASE_TREE/perf"
cp -R "$ROOT_DIR/perf" "$BASE_TREE/perf"

NATIVE_DIRS=(candle-binding ml-binding nlp-binding onnx-binding)
REUSE_NATIVE=true
if ! git -C "$ROOT_DIR" diff --quiet "$BASE_COMMIT" -- \
  "${NATIVE_DIRS[@]}" tools/make/rust.mk tools/make/common.mk tools/make/build-run-test.mk Makefile; then
  REUSE_NATIVE=false
fi
for binding in "${NATIVE_DIRS[@]}"; do
  if [[ ! -d "$ROOT_DIR/$binding/target/release" ]]; then
    REUSE_NATIVE=false
  fi
done
if [[ "$REUSE_NATIVE" == true ]]; then
  echo "Reusing native libraries: binding sources and build inputs are unchanged."
  for binding in "${NATIVE_DIRS[@]}"; do
    ln -s "$ROOT_DIR/$binding/target" "$BASE_TREE/$binding/target"
  done
else
  echo "Building base native libraries: sources or build inputs differ."
  PREBUILT_NATIVE_LIBS=0 GIT_WORK_TREE="$BASE_TREE" make -C "$BASE_TREE" rust-ci
fi

# Resolve include/link locations from the base tree. Never put current libraries
# ahead of rebuilt base libraries in the dynamic loader's search path.
BASE_NATIVE_PATH=""
BASE_LDFLAGS=""
for binding in "${NATIVE_DIRS[@]}"; do
  BASE_NATIVE_PATH="${BASE_NATIVE_PATH:+$BASE_NATIVE_PATH:}$BASE_TREE/$binding/target/release"
  BASE_LDFLAGS="$BASE_LDFLAGS -L$BASE_TREE/$binding/target/release"
done
export LD_LIBRARY_PATH="$BASE_NATIVE_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CGO_LDFLAGS="$BASE_LDFLAGS"
export CGO_ENABLED=1
(
  cd "$BASE_TREE/perf"
  export GIT_WORK_TREE="$BASE_TREE"
  if ! go test -run '^$' -bench '^(BenchmarkClassify|BenchmarkCGO|BenchmarkCache)' \
    -benchmem -benchtime="${PERF_MODEL_BENCHTIME:-3s}" -timeout=30m ./benchmarks/... \
    2>&1 | tee "$OUTPUT_DIR/model-baseline-output.txt"; then
    echo "Base revision $BASE_COMMIT cannot run the current model benchmark contract; inspect compile/runtime output and choose an explicitly compatible base." >&2
    exit 1
  fi
  go run ./cmd/perftest --parse-bench="$OUTPUT_DIR/model-baseline-output.txt" \
    --output="$OUTPUT_DIR/model-baseline.json"
)
