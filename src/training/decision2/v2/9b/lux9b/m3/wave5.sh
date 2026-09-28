#!/usr/bin/env bash
# usage: wave5.sh SHA
# Arm D seed soup once all three seed pipelines (training, CAL698, readouts) have ended: uniform
# FP32 soup on CPU, then CAL698 temperatures and typed DEV + CSS pilot readouts on GPU4.
set -uo pipefail
sha=$1
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
L=$S/v2/9b/lux9b/m3
M3=/data/dev2/runs/9b/m3
idle() {
  ! grep -q "^status=running" "/data/dev2/leases/gpu$1.lock/owner" 2>/dev/null \
    && [ "$(rocm-smi -d "$1" --showmemuse | awk -F': ' '/VRAM%/ {v=$NF} END {print v}')" = 0 ]
}
for _ in $(seq 1 1440); do
  running=0
  for pattern in "arm.sh [0-9a-f]+ 4 D-s1 " "arm.sh [0-9a-f]+ 6 D-s2 " "arm.sh [0-9a-f]+ 7 D-s3 "; do
    pgrep -f "$pattern" >/dev/null && running=1
  done
  [ "$running" = 0 ] && break
  sleep 30
done
for name in D-s1 D-s2 D-s3; do
  [ "$(cat "$M3/$name-css-pilot/exit-code.txt" 2>/dev/null)" = 0 ] || { echo "$name pipeline incomplete; no soup"; exit 1; }
done
for _ in $(seq 1 240); do idle 4 && break; sleep 15; done
idle 4 || { echo "GPU4 not idle"; exit 1; }
"$L/soup.sh" "$sha" 4 D-soup D-s1 D-s2 D-s3
