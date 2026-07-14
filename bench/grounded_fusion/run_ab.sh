#!/usr/bin/env bash
# End-to-end A/B driver for the grounding-aware fusion DRACO benchmark.
#
# Brings the router up with the grounding-ON config, runs the `on` arm, restarts
# with the grounding-OFF config, runs the `off` arm, then prints the paired report.
# Assumes Ollama, the no-think proxy (:11435), and Envoy (:8801) are already up
# (this script will prep+start Envoy if it isn't).
#
# Usage:
#   bench/grounded_fusion/run_ab.sh --max-samples 8 --domains Medicine,Law --grade-panel
# Extra args are forwarded to evaluate.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY=.venv-bench/bin/python
DRACO="${DRACO_PATH:-$HOME/Downloads/draco.json}"
EXTRA_ARGS=("$@")
export LD_LIBRARY_PATH="$REPO/candle-binding/target/release:$REPO/ml-binding/target/release:$REPO/nlp-binding/target/release"

start_router() {  # $1 = config path
  pkill -f "bin/router" 2>/dev/null || true; sleep 2
  nohup ./bin/router -config="$1" > /tmp/router_ab.log 2>&1 &
  echo "  router starting ($1)..."
  for _ in $(seq 1 40); do
    if grep -q "hallucination_nli_initialized" /tmp/router_ab.log 2>/dev/null \
       && lsof -nP -iTCP:50051 -sTCP:LISTEN >/dev/null 2>&1; then
      echo "  router ready (NLI wired, extproc :50051)"; sleep 2; return 0
    fi
    sleep 2
  done
  echo "  ERROR: router did not become ready; see /tmp/router_ab.log"; exit 1
}

ensure_envoy() {
  if lsof -nP -iTCP:8801 -sTCP:LISTEN >/dev/null 2>&1; then return; fi
  echo "Preparing + starting Envoy on :8801..."
  sed '80,89d' deploy/local/envoy.yaml | sed 's/300s/1800s/g' > /tmp/envoy-bench.yaml
  nohup tools/bin/func-e run --config-path /tmp/envoy-bench.yaml > /tmp/envoy.log 2>&1 &
  sleep 12
}

ensure_proxy() {
  if lsof -nP -iTCP:11435 -sTCP:LISTEN >/dev/null 2>&1; then return; fi
  echo "Starting no-think Ollama proxy on :11435..."
  nohup $PY -m bench.grounded_fusion.ollama_proxy --port 11435 > /tmp/proxy.log 2>&1 &
  sleep 3
}

echo "== regenerating configs =="
$PY -m bench.grounded_fusion.make_configs >/dev/null
ensure_proxy
ensure_envoy

echo "== ARM: on (grounding enabled) =="
start_router bench/grounded_fusion/config-fusion-on.yaml
$PY -m bench.grounded_fusion.evaluate --endpoint http://localhost:8801 \
  --draco-path "$DRACO" --arm on --assert-grounding "${EXTRA_ARGS[@]}"

echo "== ARM: off (plain fusion) =="
start_router bench/grounded_fusion/config-fusion-off.yaml
$PY -m bench.grounded_fusion.evaluate --endpoint http://localhost:8801 \
  --draco-path "$DRACO" --arm off "${EXTRA_ARGS[@]}"

echo "== A/B report =="
$PY -m bench.grounded_fusion.compare --results-dir bench/grounded_fusion/results \
  --json-out bench/grounded_fusion/results/ab_report.json

echo "Done. Teardown: pkill -f 'bin/router'; pkill -f func-e; pkill -f envoy; pkill -f ollama_proxy"
