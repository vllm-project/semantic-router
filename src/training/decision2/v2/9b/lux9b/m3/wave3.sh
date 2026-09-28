#!/usr/bin/env bash
# usage: wave3.sh SHA
# Milestone 3 arm D (E8F-style: Lux 1.0 full fine-tuning on full-M + A7 + v1, own-Lux KL on
# recipe rows), three seeds. GPU4: seed 20260926 with preflights once the same-renderer Lux1
# 16K control has finished there. GPU6 / GPU7: seeds 1 / 2 once arm A's pipeline on that GPU
# has ended and the preflight has passed.
set -uo pipefail
sha=$1
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
L=$S/v2/9b/lux9b/m3
M3=/data/dev2/runs/9b/m3
FULL=(--train-mode full --backbone-lr 1e-5 --head-lr 1e-4 --max-batch-tokens 32768 --max-batch-rows 64
  --update-rows 64)
idle() {
  ! grep -q "^status=running" "/data/dev2/leases/gpu$1.lock/owner" 2>/dev/null \
    && [ "$(rocm-smi -d "$1" --showmemuse | awk -F': ' '/VRAM%/ {v=$NF} END {print v}')" = 0 ]
}
wait_process_then_idle() {
  gpu=$1; pattern=$2; log=$3
  for _ in $(seq 1 1440); do
    if [ -s "$log" ] && ! pgrep -f "$pattern" >/dev/null && idle "$gpu"; then return 0; fi
    sleep 15
  done
  return 1
}
preflight_status() {
  for _ in $(seq 1 480); do
    for step in zero one check; do
      code=$(cat "$M3/pf-D-s1-$step/exit-code.txt" 2>/dev/null || true)
      [ -n "$code" ] && [ "$code" != 0 ] && { echo "FAIL"; return; }
    done
    if [ -f "$M3/pf-D-s1-check/exit-code.txt" ]; then
      python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M3/pf-D-s1-check/preflight.json"
      return
    fi
    sleep 15
  done
  echo "TIMEOUT"
}
(
  wait_process_then_idle 4 "formal.sh [0-9a-f]+ 4 lux1" "$M3/logs/formal-lux1.log" || exit 1
  "$L/arm.sh" "$sha" 4 D-s1 20260926 d-full --preflight "${FULL[@]}"
) > "$M3/logs/D-s1.log" 2>&1 &
for pair in "6 A-s1 1 D-s2" "7 A-s2 2 D-s3"; do
  read -r gpu prior seed name <<< "$pair"
  (
    wait_process_then_idle "$gpu" "arm.sh [0-9a-f]+ $gpu $prior " "$M3/logs/$prior.log" || exit 1
    status=$(preflight_status)
    [ "$status" = PASS ] || { echo "preflight status '$status'; $name not started"; exit 1; }
    "$L/arm.sh" "$sha" "$gpu" "$name" "$seed" d-full "${FULL[@]}"
  ) > "$M3/logs/$name.log" 2>&1 &
done
wait
echo WAVE3-DONE
