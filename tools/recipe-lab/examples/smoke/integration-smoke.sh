#!/bin/bash
# recipe-lab V2 integration smoke: real components, real evidence.
#
#   R1  hot reload x real router: threshold candidates hot-swap through the
#       management API with no restart fallback and a <= 5 s switch median,
#       and rank identically under the restart runner
#   R2  verify x live sr-bench: three AB/BA repetitions with the reference
#       single-model arm constructed automatically from the target registry
#       (fault injection -> INCONCLUSIVE; dataset mismatch -> rejected)
#
# The pytest suite is self-contained (mock backends and routers are managed
# by its fixtures); this script only stands up the persistent sr-bench
# service and hands its URL to the suite via RECIPE_LAB_SRBENCH_URL.
#
# Run on a GPU node (the router loads the decision model):
#   RECIPE_LAB_ROUTER_BIN=$SR_BUILD/bin/semantic-router \
#   HF_HOME=$WORK/hf-home \
#     bash "$REPO/tools/recipe-lab/examples/smoke/integration-smoke.sh"
# Environment (all optional; defaults derive from this script's location):
#   REPO                  semantic-router checkout (default: derived)
#   RECIPE_LAB_BIN        venv bin with recipe-lab, vllm-sr and pytest
#                         (default: $REPO/.venv/bin)
#   ROUTER_BIN / SR_BUILD router binary (default: $SR_BUILD/bin/semantic-router)
#   HF_HOME               HuggingFace cache for router processes
#   SRBENCH_PORT          sr-bench service port     (default: 18099)
#   WORK_ROOT             scratch root              (default: $REPO/.tmp-smoke-runs)
set -euo pipefail

ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-smoke-runs}"
SRBENCH_PORT="${SRBENCH_PORT:-18099}"
WORK="$WORK_ROOT/integration-$(date +%m%d%H%M%S)"
mkdir -p "$WORK/srbench-store"
echo "INTEGRATION-WORK=$WORK"

[ -x "$ROUTER_BIN" ] || { echo "SMOKE-FAILED router binary missing: $ROUTER_BIN"; exit 1; }
[ -x "$RL/vllm-sr" ] || { echo "SMOKE-FAILED vllm-sr missing in $RL"; exit 1; }
"$RL/python" -c "import pytest" 2>/dev/null \
  || { echo "SMOKE-FAILED pytest missing in $RL"; exit 1; }
# The real router execs vllm-srun to serve its decision model; without it
# every router process dies at startup with "runtime command cannot run".
command -v vllm-srun > /dev/null 2>&1 \
  || { echo "SMOKE-FAILED vllm-srun not on PATH (router model-runtime serve command)"; exit 1; }

SRBENCH_PID=""
cleanup() {
  if [ -n "$SRBENCH_PID" ]; then kill "$SRBENCH_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

# ---------------- Stage 1: persistent sr-bench service ----------------
( cd "$REPO/src/vllm-sr" && exec "$RL/python" -m cli.sr_bench.service \
    --store "$WORK/srbench-store" --port "$SRBENCH_PORT" ) \
  > "$WORK/srbench-serve.log" 2>&1 &
SRBENCH_PID=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${SRBENCH_PORT}/health" > /dev/null 2>&1 && break
  kill -0 "$SRBENCH_PID" 2>/dev/null \
    || { echo "SMOKE-FAILED sr-bench died"; tail -30 "$WORK/srbench-serve.log"; exit 1; }
  sleep 1
done
curl -sf "http://127.0.0.1:${SRBENCH_PORT}/health" > /dev/null \
  || { echo "SMOKE-FAILED sr-bench not ready"; tail -30 "$WORK/srbench-serve.log"; exit 1; }
echo "SRBENCH-READY http://127.0.0.1:${SRBENCH_PORT}"

# ---------------- Stage 2: pytest integration suite (R1 + R2) ----------------
export PATH="$RL:$PATH"
export RECIPE_LAB_ROUTER_BIN="$ROUTER_BIN"
export RECIPE_LAB_SRBENCH_URL="http://127.0.0.1:${SRBENCH_PORT}"
export RECIPE_LAB_SRBENCH_STORE="$WORK/srbench-store"
export RECIPE_LAB_SRBENCH_BIN="$RL/vllm-sr"
if [ -n "${HF_HOME:-}" ]; then
  export RECIPE_LAB_HF_HOME="$HF_HOME"
fi

cd "$REPO/tools/recipe-lab"
set +e
"$RL/python" -m pytest tests/integration -m integration -v
RC=$?
set -e

echo "INTEGRATION-DONE rc=$RC work=$WORK"
exit "$RC"
