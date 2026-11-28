#!/bin/bash
# Probe the router's 400 with sr-bench-style streaming requests.
#
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
WORK="$WORK_ROOT/probe-400-$(date +%H%M%S)"
mkdir -p "$WORK"
"$PYBIN" "$ASSETS/mock_backend.py" --port 8000 > "$WORK/mock.log" 2>&1 &
MOCK_PID=$!
sleep 1
"$ROUTER_BIN" -config "$ASSETS/smoke-config.yaml" -gateway standalone > "$WORK/router.log" 2>&1 &
ROUTER_PID=$!
for i in $(seq 1 60); do
  sleep 5
  kill -0 $ROUTER_PID 2>/dev/null || { tail -20 "$WORK/router.log"; exit 1; }
  curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true' && break
done
"$RL/python" "$ASSETS/probe_400.py" --only-mom-headers --config-hash "$(curl -sf http://127.0.0.1:8080/api/v1/config/hash 2>/dev/null | grep -o "[0-9a-f]\{64\}" | head -1)" 2>&1
echo "PROBE-DONE $WORK"
kill $ROUTER_PID $MOCK_PID 2>/dev/null
