#!/usr/bin/env bash
# 27B MoE Stage B, node B side (host; detached): the preregistered steps from the relayed seeds to the verdicts
# (amendment 4). Waits for both relayed BEST checkpoints (moe-stageb-nodeA.sh), then moe-tail.sh soup, readout (T = 1,
# node B GPU7), cal698, adopt (23:15 rule) and devgates. A soup that passes the development gates gets package (frozen
# before any formal collection), formal (smoke, collect, score), gates, latency and mlx (staged for node A), then the
# chain waits up to 4 h for node A's mlx-diag pairing (mlx-pull) and writes the verdicts. Before each GPU stage the
# milestone total (node A's relayed receipts + node B's) plus the stage's plan must stay within 60 GPU-h. Any failed
# stage stops the chain; on every exit without a pushed mlx-diag collection, X/mlx/NAME.SKIP tells node A.
# Usage: moe-stageb-nodeB.sh MIRROR [NAME]
set -euo pipefail
MIR=$1 NAME=${2:-MOE-Git-soup}
S=/data/dev2/src/$MIR/src/training/decision2
R=/data/dev2/runs/27b-moe
X=/data/dev2/xfer/27b-moe
TAIL=$S/v2/27b/moe/moe-tail.sh
ARMS=(MOE-Git-s1 MOE-Git-s2)
CAP=60
log() { echo "$(date -u +%FT%TZ) $*"; }
skip() {
  mkdir -p "$X/mlx"
  [ -f "$X/mlx/$NAME.SKIP" ] || echo "$1" > "$X/mlx/$NAME.SKIP"
  log "no mlx-diag collection for $NAME: $1"
}
on_exit() {
  local code=$?
  [ -f "$X/mlx/$NAME.PUSHED" ] || skip "node B chain ended (exit $code) before the mlx-diag collection"
}
trap on_exit EXIT
budget() {  # PLANNED_GPU_HOURS
  python3 - "$R" "$X/relay/BUDGET-nodeA.json" "$1" "$CAP" <<'EOF'
import glob, json, sys
root, node_a, planned, cap = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
paths = [*glob.glob(f"{root}/*/receipts/*.json"), *glob.glob(f"{root}/pathcheck/*/receipts/*.json"),
         *glob.glob(f"{root}/readouts/*/GPU-TIME.json"), *glob.glob(f"{root}/formal/*/GPU-TIME.json")]
b = sum(json.load(open(p)).get("gpu_hours", 0) for p in paths)
a = json.load(open(node_a))["gpu_hours"]
print(f"budget: node A {a:.3f} + node B {b:.3f} = {a + b:.3f} GPU-h; next stage plans {planned}; cap {cap}")
sys.exit(0 if a + b + planned <= cap else 1)
EOF
}
[ -f "$TAIL" ] || { log "missing mirror $MIR"; exit 2; }
ready() {
  for arm in "${ARMS[@]}"; do [ -f "$X/relay/$arm-best/RELAYED" ] || return 1; done
  [ -f "$X/relay/BUDGET-nodeA.json" ]
}
until [ -f "$X/relay/STAGEB-ABSENT" ] || ready; do sleep 300; done
if [ -f "$X/relay/STAGEB-ABSENT" ]; then
  skip "seed(s) ended without COMPLETE.json: $(cat "$X/relay/STAGEB-ABSENT")"
  exit 0
fi
log "both BEST checkpoints relayed"
bash "$TAIL" soup "$MIR" "$NAME" "${ARMS[@]}"
budget 0.8
bash "$TAIL" readout "$MIR" "$NAME"
bash "$TAIL" cal698 "$MIR" "$NAME"
bash "$TAIL" adopt "$MIR" "$NAME"
bash "$TAIL" devgates "$MIR" "$NAME"
if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2] in d['finalists'] else 3)" \
  "$R/readouts/DEVGATES.json" "$NAME"; then
  skip "development gates failed (readouts/DEVGATES.json)"
  exit 0
fi
bash "$TAIL" package "$MIR" "$NAME"
budget 3.0
bash "$TAIL" formal "$MIR" "$NAME"
bash "$TAIL" gates "$MIR" "$NAME"
budget 0.8
bash "$TAIL" latency "$MIR" "$NAME"
bash "$TAIL" mlx "$MIR" "$NAME"
log "waiting for node A's mlx-diag pairing"
for _ in $(seq 48); do
  [ -f "$X/mlx/$NAME-vs-A20r.json" ] && break
  sleep 300
done
if [ -f "$X/mlx/$NAME-vs-A20r.json" ]; then
  bash "$TAIL" mlx-pull "$MIR" "$NAME"
else
  log "no mlx-diag pairing after 4 h: item 4 stays PENDING"
fi
bash "$TAIL" verdicts "$MIR" "$NAME"
log "stage B node B done"
