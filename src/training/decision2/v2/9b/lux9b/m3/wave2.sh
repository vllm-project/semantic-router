#!/usr/bin/env bash
# usage: wave2.sh SHA
# Milestone 3 wave 2 on node A GPU2-4 once the research & data track has released them (lease
# not running, no VRAM in use). GPU2: arm B (AutoJev-27B targets, same TRAIN as A), primary seed,
# with preflights. GPU3: arm B seed 1 after the preflight passes. GPU4: the same-renderer Lux1
# 16K formal control, from a frozen copy of the Milestone 3 autotune cache.
set -uo pipefail
sha=$1
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
L=$S/v2/9b/lux9b/m3
M3=/data/dev2/runs/9b/m3
F=/data/dev2/runs/9b/formal-m3
free() {
  for _ in $(seq 1 720); do
    if ! grep -q "^status=running" "/data/dev2/leases/gpu$1.lock/owner" 2>/dev/null \
      && [ "$(rocm-smi -d "$1" --showmemuse | awk -F': ' '/VRAM%/ {v=$NF} END {print v}')" = 0 ]; then
      return 0
    fi
    sleep 10
  done
  return 1
}
preflight_status() {
  for _ in $(seq 1 360); do
    for step in zero one check; do
      code=$(cat "$M3/pf-$1-$step/exit-code.txt" 2>/dev/null || true)
      [ -n "$code" ] && [ "$code" != 0 ] && { echo "FAIL"; return; }
    done
    if [ -f "$M3/pf-$1-check/exit-code.txt" ]; then
      python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M3/pf-$1-check/preflight.json"
      return
    fi
    sleep 10
  done
  echo "TIMEOUT"
}
(
  free 2 || exit 1
  "$L/arm.sh" "$sha" 2 B-s1 20260926 b-full-M --preflight
) > "$M3/logs/B-s1.log" 2>&1 &
(
  free 3 || exit 1
  status=$(preflight_status B-s1)
  [ "$status" = PASS ] || { echo "preflight status '$status'; seed 1 not started"; exit 1; }
  "$L/arm.sh" "$sha" 3 B-s2 1 b-full-M
) > "$M3/logs/B-s2.log" 2>&1 &
(
  free 4 || exit 1
  mkdir -p "$F"
  [ -e "$F/triton-cache" ] || cp -a "$M3/triton-cache" "$F/triton-cache"
  printf "track=9b-clm\nstatus=idle\nclaimed_utc=%s\n" "$(date -u +%FT%TZ)" > /data/dev2/leases/gpu4.lock/owner
  "$L/formal.sh" "$sha" 4 lux1
) > "$M3/logs/formal-lux1.log" 2>&1 &
wait
echo WAVE2-DONE
