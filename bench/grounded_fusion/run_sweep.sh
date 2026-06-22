#!/usr/bin/env bash
# Threshold-sweep A/B for grounding-aware fusion on the contested domains.
#
# Grounding's min_score only changes FILTER+SYNTHESIS; the panel responses, their
# NLI grounding scores, and the panel rubric grades are threshold-independent, so:
#   - the OFF arm runs once,
#   - the panel is graded once (on the first/base threshold),
#   - the ON arm runs once per threshold.
# Each ON arm is compared against the single OFF arm (MVP two-config paired A/B at
# temperature 0; panels are regenerated per router run, so cross-threshold deltas
# carry some resampling noise — report the trend, not 4th-decimal differences).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY=.venv-bench/bin/python
DRACO="${DRACO_PATH:-$HOME/Downloads/draco.json}"
DOMAINS="${DOMAINS:-Medicine,Law}"
THRESHOLDS=(0.34 0.55 0.60)
RESBASE=bench/grounded_fusion/results_sweep
export LD_LIBRARY_PATH="$REPO/candle-binding/target/release:$REPO/ml-binding/target/release:$REPO/nlp-binding/target/release"

start_router() {  # $1 = config path
  pkill -f "bin/router" 2>/dev/null || true; sleep 3
  nohup ./bin/router -config="$1" > /tmp/router_ab.log 2>&1 &
  for _ in $(seq 1 40); do
    if grep -q "hallucination_nli_initialized" /tmp/router_ab.log 2>/dev/null \
       && lsof -nP -iTCP:50051 -sTCP:LISTEN >/dev/null 2>&1; then
      echo "  router ready ($1)"; sleep 2; return 0
    fi
    sleep 2
  done
  echo "  ERROR: router not ready ($1)"; tail -15 /tmp/router_ab.log; exit 1
}

ensure_support() {
  lsof -nP -iTCP:11435 -sTCP:LISTEN >/dev/null 2>&1 || { echo "starting proxy"; nohup $PY -m bench.grounded_fusion.ollama_proxy --port 11435 >/tmp/proxy.log 2>&1 & sleep 3; }
  lsof -nP -iTCP:8801 -sTCP:LISTEN >/dev/null 2>&1 || { echo "ERROR: envoy :8801 down"; exit 1; }
}

echo "== regenerating base configs =="
$PY -m bench.grounded_fusion.make_configs >/dev/null
ensure_support
mkdir -p "$RESBASE/off"

echo "== OFF arm (once) =="
start_router bench/grounded_fusion/config-fusion-off.yaml
$PY -m bench.grounded_fusion.evaluate --endpoint http://localhost:8801 \
  --draco-path "$DRACO" --arm off --domains "$DOMAINS" --resume \
  --output-dir "$RESBASE/off"

first=1
for T in "${THRESHOLDS[@]}"; do
  echo "== ON arm  min_score=$T =="
  cfg="bench/grounded_fusion/config-fusion-on-$T.yaml"
  sed "s/min_score: 0.34/min_score: $T/" bench/grounded_fusion/config-fusion-on.yaml > "$cfg"
  outdir="$RESBASE/on_$T"; mkdir -p "$outdir"
  start_router "$cfg"
  GP=""; [ "$first" = "1" ] && GP="--grade-panel"   # Level-1 is threshold-independent
  $PY -m bench.grounded_fusion.evaluate --endpoint http://localhost:8801 \
    --draco-path "$DRACO" --arm on --assert-grounding --domains "$DOMAINS" $GP --resume \
    --output-dir "$outdir"
  first=0
  cp "$RESBASE/off/samples_off.jsonl" "$RESBASE/off/summary_off.json" "$outdir/"
  echo "-- A/B report (min_score=$T) --"
  $PY -m bench.grounded_fusion.compare --results-dir "$outdir" \
    --json-out "$outdir/ab_report.json"
done

echo "== SWEEP DONE =="
