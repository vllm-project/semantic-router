#!/bin/bash
# Agentic-traffic probe (Sub-issue 6 groundwork): multi-turn conversation +
# tool-calling payloads through the router with session protection enabled.
# CPU-only; runs on the head node or a compute node.
# Environment (all optional; defaults derive from this script's location):
#   REPO            semantic-router checkout        (default: derived)
#   RECIPE_LAB_BIN  dir with recipe-lab CLI+python  (default: $REPO/.venv/bin)
#   PYTHON_BIN      python for the mock backend     (default: $RECIPE_LAB_BIN/python)
#   SR_BUILD        build prefix                    (default: $REPO/build)
#   ROUTER_BIN      router binary                   (default: $SR_BUILD/bin/semantic-router)
#   WORK_ROOT       scratch root for run dirs       (default: $REPO/.tmp-smoke-runs)
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
PYBIN="${PYTHON_BIN:-$RL/python}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-smoke-runs}"
W="$WORK_ROOT/agent-probe-$(date +%m%d%H%M%S)"
mkdir -p "$W"
echo "PROBE-WORK=$W"

ROUTER_PID=""; MOCK_PID=""
cleanup() { for pid in ${ROUTER_PID:-} ${MOCK_PID:-}; do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

"$PYBIN" "$ASSETS/mock_backend.py" --port 8000 > "$W/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo PROBE-FAILED mock; cat "$W/mock.log"; exit 1; }

"$ROUTER_BIN" -config "$ASSETS/agent-probe-config.yaml" -gateway standalone > "$W/router.log" 2>&1 &
ROUTER_PID=$!
READY=0
for i in $(seq 1 60); do
  sleep 5
  kill -0 $ROUTER_PID 2>/dev/null || { echo PROBE-FAILED router-died; tail -30 "$W/router.log"; exit 1; }
  if curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true'; then
    READY=1; echo "ROUTER-READY after ~$((i*5))s"; break
  fi
done
[ "$READY" = "1" ] || { echo PROBE-FAILED not-ready; tail -30 "$W/router.log"; exit 1; }

"$RL/python" "$ASSETS/agentic_probe.py" --output-dir "$W" 2>&1 | tee "$W/probe-run.log"
echo "PROBE-DONE $W"
