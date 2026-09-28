#!/usr/bin/env bash
# usage: wave4.sh SHA
# Milestone 3 arm DL (arm D's TRAIN with arm A's LoRA recipe). GPU3: seed 20260926 with
# preflights once arm B's pipeline on GPU3 has ended. GPU2: seed 1 once arm B's pipeline on
# GPU2 has ended, the file logs/DL-s2.go exists (written after any arm B formal run on GPU2)
# and the preflight has passed.
set -uo pipefail
sha=$1
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
L=$S/v2/9b/lux9b/m3
M3=/data/dev2/runs/9b/m3
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
      code=$(cat "$M3/pf-DL-s1-$step/exit-code.txt" 2>/dev/null || true)
      [ -n "$code" ] && [ "$code" != 0 ] && { echo "FAIL"; return; }
    done
    if [ -f "$M3/pf-DL-s1-check/exit-code.txt" ]; then
      python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M3/pf-DL-s1-check/preflight.json"
      return
    fi
    sleep 15
  done
  echo "TIMEOUT"
}
(
  wait_process_then_idle 3 "arm.sh [0-9a-f]+ 3 B-s2 " "$M3/logs/B-s2.log" || exit 1
  "$L/arm.sh" "$sha" 3 DL-s1 20260926 d-full --preflight
) > "$M3/logs/DL-s1.log" 2>&1 &
(
  wait_process_then_idle 2 "arm.sh [0-9a-f]+ 2 B-s1 " "$M3/logs/B-s1.log" || exit 1
  for _ in $(seq 1 1440); do [ -f "$M3/logs/DL-s2.go" ] && break; sleep 15; done
  [ -f "$M3/logs/DL-s2.go" ] || exit 1
  wait_process_then_idle 2 "formal.sh [0-9a-f]+ 2 " "$M3/logs/DL-s2.go" || exit 1
  status=$(preflight_status)
  [ "$status" = PASS ] || { echo "preflight status '$status'; DL-s2 not started"; exit 1; }
  "$L/arm.sh" "$sha" 2 DL-s2 1 d-full
) > "$M3/logs/DL-s2.log" 2>&1 &
wait
echo WAVE4-DONE
