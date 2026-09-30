#!/usr/bin/env bash
# ~27B M5 branch B1 post-training chain on node B (host side; preregistration amendment 2). Waits for L128-s1 (node B)
# and the L128-s2 relay (node A, m5-l128-relay.sh), pulls L128-s2 (m5-pull.sh), builds the exact rank-256 soup M5-L128
# of the two BEST adapters (m5-tail.sh lsoup; node A's member checked against its relay list), reads it out on GPU
# (m5-tail.sh readout, CHECKPOINT_FORMAT=peft-lora/1: CAL698 kernel fit, then typed DEV + CSS pilot + HT-DEV v2 on
# fresh copies of 03b172f1) and runs the development gates over every M5 candidate with a readout (A20r reference;
# the proxy pool holds all of them). The gates decide:
#   - not a finalist: the chain records it, leaves mlx/M5-L128.SKIP for node A's watcher, and ends;
#   - finalist: if the receipts on both nodes plus RESERVE (formal + mlx-diag) stay within 72 GPU-h, the formal run
#     (m5-tail.sh formal; LOADED_PARAMETERS 27,497,508,864; sealed FF finalists as comparators), the mlx-diag
#     collection and its push to node A, where m5-l128-relay.sh scores and pairs it; then mlx-pull and the host-CPU
#     gates (m5-gates.sh gates, overlap, verdicts). Past the budget it stops before the formal run with a record.
# A stopped seed or a failed stage ends the chain with a record; nothing reruns.
# Usage: m5-l128-chain.sh MIRROR_SHA GPU S1_DRIVER_PID     (GPU: node B GPU0-2, a track-27b lease)
set -euo pipefail
echo "m5 l128 chain $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} GPU=${2:?GPU} PID=${3:?S1_DRIVER_PID}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
case "$GPU" in 0 | 1 | 2) ;; *) echo "GPU$GPU is not an M5 auxiliary GPU (node B GPU0-2)" >&2; exit 2 ;; esac
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "S1_DRIVER_PID must be a pid" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
TAIL=$S/v2/27b/m5/m5-tail.sh GATES=$S/v2/27b/m5/m5-gates.sh
[ -f "$S/v2/27b/m5/m5-l128-chain.sh" ] || { echo "missing mirror $SHA" >&2; exit 2; }
R=/data/dev2/runs/27b/m5 S1=/data/dev2/runs/27b/M5-L128-s1
KEY=/data/dev2/tmp/27b-m5-xfer
LOADED=27497508864 CAP=72 RESERVE=${RESERVE:-1.0}
X="ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
stamp() { date -u +%FT%TZ; }
on_a() {  # PATH under node A's relay root exists
  rsync --list-only -e "$X" "root@$(cat "$KEY/peer"):$1" > /dev/null 2>&1
}
skip_mlx() {  # REASON: tell node A's watcher that no mlx-diag collection will come
  printf '%s %s\n' "$(stamp)" "$1" > "$R/logs/M5-L128.SKIP"
  rsync -a --mkpath -e "$X" "$R/logs/M5-L128.SKIP" "root@$(cat "$KEY/peer"):mlx/M5-L128.SKIP"
}

s1=no
while :; do
  if [ "$s1" = no ] && [ -f "$S1/full/RUN_DIR" ]; then
    s1=yes
  elif [ "$s1" = no ] && ! kill -0 "$PID" 2> /dev/null; then
    sleep 10
    if [ -f "$S1/full/RUN_DIR" ]; then s1=yes; else
      echo "$(stamp) L128-s1 ended without a finished run: no soup, nothing reruns"
      tail -n 20 "$S1/driver.log"
      skip_mlx "L128-s1 ended without a finished run"
      exit 3
    fi
  fi
  if on_a relay/M5-L128-s2/RELAY-FAILED.txt; then
    echo "$(stamp) L128-s2 ended without a finished run on node A (relay RELAY-FAILED.txt): no soup, nothing reruns"
    skip_mlx "L128-s2 ended without a finished run"
    exit 3
  fi
  [ "$s1" = yes ] && on_a relay/M5-L128-s2/SHA256SUMS && break
  sleep 300
done
echo "$(stamp) L128 seeds ready"
bash "$S/v2/27b/m5/m5-pull.sh" M5-L128-s2
C1=$(python3 - "$S1/full" <<'EOF'
import json, pathlib, sys
full = pathlib.Path(sys.argv[1])
run = full / (full / "RUN_DIR").read_text().strip()
best = json.loads((run / "BEST.json").read_text())["checkpoint"]
complete = json.loads((run / "COMPLETE.json").read_text())
if complete.get("status") != "complete" or complete.get("best") != best:
    raise SystemExit(f"{run} is not complete with a frozen BEST")
print(run / best)
EOF
)
C2=$R/relay/M5-L128-s2/checkpoint
echo "$(stamp) soup M5-L128 from $C1 and $C2"
[ -f "$R/M5-L128/checkpoint/soup_manifest.json" ] ||
  RELAY_SUMS="$C2=$R/relay/M5-L128-s2/SHA256SUMS.nodeB" bash "$TAIL" lsoup "$SHA" M5-L128 "$C1" "$C2"
export CHECKPOINT_FORMAT=peft-lora/1
[ -f "$R/readouts/M5-L128/READOUT-M4B.json" ] || bash "$TAIL" readout "$SHA" M5-L128 "$R/M5-L128/checkpoint" "$GPU"
names=()
for n in M5-FF20 M5-FF20H M5-SX M5-L128; do [ -f "$R/readouts/$n/READOUT-M4B.json" ] && names+=("$n"); done
bash "$TAIL" devgates "$SHA" "${names[@]}"
DEVGATES=$(find "$R/readouts" -maxdepth 1 -name 'DEVGATES-*.json' | sort | tail -n 1)
verdict=$(python3 - "$DEVGATES" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
c = d["candidates"]["M5-L128"]
print("finalist" if "M5-L128" in d["finalists"] else "not a finalist: " + ", ".join(
    k for k, g in c["gates"].items() if not g["pass"]))
EOF
)
echo "$(stamp) M5-L128 development gates ($DEVGATES): $verdict"
if [ "$verdict" != finalist ]; then
  skip_mlx "M5-L128 $verdict"
  echo "m5 l128 chain complete (no formal run): $(stamp)"
  exit 0
fi
total=$(python3 - "$R/relay/M5-L128-s2/BUDGET-nodeA.json" <<'EOF'
import glob, json, os, sys
seen, total = set(), 0.0
paths = glob.glob("/data/dev2/runs/27b/m5/**/*.json", recursive=True)
paths += glob.glob("/data/dev2/runs/27b/M5-L128-s*/**/*.json", recursive=True)
for path in paths:
    real = os.path.realpath(path)
    if real in seen or "/relay/" in path or "/triton-cache" in path:
        continue
    seen.add(real)
    try:
        record = json.load(open(path))
    except Exception:
        continue
    if isinstance(record, dict) and (os.path.basename(path) == "GPU-TIME.json" or (
            os.path.basename(os.path.dirname(path)) == "receipts" and "gpu_hours" in record)):
        total += float(record.get("gpu_hours", 0))
print(round(total + json.load(open(sys.argv[1]))["gpu_hours"], 3))
EOF
)
echo "$(stamp) M5 receipts on both nodes: $total GPU-h; formal + mlx-diag reserve $RESERVE; cap $CAP"
if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) + float(sys.argv[2]) <= float(sys.argv[3]) else 1)' \
  "$total" "$RESERVE" "$CAP"; then
  echo "$(stamp) budget: $total + $RESERVE passes $CAP GPU-h; M5-L128's formal run waits for a coordinator decision"
  exit 4
fi
extra=()
for n in M5-FF20H M5-SX; do [ -f "$R/$n/formal/SEAL.json" ] && extra+=("$n=$R/$n/formal"); done
[ -f "$R/M5-L128/formal/SEAL.json" ] ||
  LOADED_PARAMETERS=$LOADED EXTRA_COMPARATOR="${extra[*]}" bash "$TAIL" formal "$SHA" M5-L128 "$R/M5-L128/checkpoint" "$GPU"
[ -f "$R/mlx-diag/M5-L128/COLLECT.json" ] || bash "$TAIL" mlx "$SHA" M5-L128 "$GPU"
bash "$TAIL" mlx-push "$SHA" M5-L128
stamp > "$R/logs/M5-L128.PUSHED"
rsync -a -e "$X" "$R/logs/M5-L128.PUSHED" "root@$(cat "$KEY/peer"):mlx/M5-L128.PUSHED"
until on_a mlx/M5-L128-vs-A20r.json; do sleep 120; done
sleep 60
bash "$TAIL" mlx-pull "$SHA" M5-L128
finalists=(M5-L128)
for n in M5-FF20H M5-SX; do [ -f "$R/$n/formal/SEAL.json" ] && finalists+=("$n"); done
bash "$GATES" "$SHA" gates "${finalists[@]}"
bash "$GATES" "$SHA" overlap "${finalists[@]}"
bash "$GATES" "$SHA" verdicts "${finalists[@]}"
echo "m5 l128 chain complete: $(stamp)"
