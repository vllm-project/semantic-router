#!/usr/bin/env bash
# Decoder M10 formal collections (prereg "Formal and successor"): the M6 formal library on node E / F
# (M6_4B_NODE=E|F: image dbe5f32b, copies of node B's frozen masters formal/m5/cache-frozen{,-mlx}, isolated runner),
# outputs under /data/dev2/runs/dec/formal/m10. Points come from m10/select/formal/4b-finalists.json
# (m10_formal_select.py); 4b-C0 (DEV2.0-4B's weights) is collected first as the formal-path parity run against the
# stored bar run. For each point in order: m6-formal.sh 4b smoke <point> 8, then m6-formal.sh 4b finalist <point>
# (stage: CAL698 16K fit + 23:15 rule, package list, parameter stubs; then typed FINAL, CSS15 and public 231 at
# 16,384 tokens on a copy of the master cache, gold-free seal). mlx-diag follows after the node-A report
# (m6-formal.sh 4b mlx <run>). A failed step stops that point (never rerun); scoring is on node A.
#
# usage: M10_NODE=e|f m10-formal.sh launch|run <mirror-dir> <gpu> <point> [<point> ...]
set -u
MODE=$1 SRC=$2 GPU=$3
shift 3
NODE=${M10_NODE:?set M10_NODE=e or f}
M=/data/dev2/runs/dec/m10
F=/data/dev2/runs/dec/formal/m10
S=/data/dev2/src/$SRC/src/training/decision2
mkdir -p "$F/logs" "$F/status"
TAG=$NODE$GPU
if [ "$MODE" = launch ]; then
  mkdir "$F/logs/formal-$TAG.lock" 2> /dev/null || { echo "M10 formal $TAG already launched"; exit 0; }
  M10_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$GPU" "$@" > "$F/logs/formal-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$F/logs/formal-$TAG.pid"
  echo "$(date -u +%FT%TZ) M10 formal $TAG launched for $* (pid $(cat "$F/logs/formal-$TAG.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) formal-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
export M6_4B_NODE=${NODE^^} M6_FORMAL_ROOT=$F M6_SELECT=$M/select/formal M6_PREFIX=m10 M6_GPU=$GPU
# launch.sh (the CAL698 fit) reads the node's decoder cache for dbe5f32b: seed it from node B's (as M10's own).
T=/data/dev2/runs/dec/triton-cache/dbe5f32b2263
[ -d "$T" ] || cp -a "$M/inputs/triton-T0-dbe5f32b2263" "$T"
lease=/data/dev2/leases/gpu$GPU.lock/owner
printf 'track=dec-m10\nstatus=busy\npurpose=decoder M10 formal %s (runner entries owner.dec-formal)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$*" "$(date -u +%FT%TZ)" "$(date -u -d '+120 min' +%FT%TZ)" > "$lease"
for point in "$@"; do
  run=m10-$point
  if [ ! -f "$F/status/$run.SMOKE" ]; then
    if bash "$S/v2/dec/ops/m6/m6-formal.sh" 4b smoke "$point" 8 > "$F/logs/$run-smoke.log" 2>&1; then
      echo "done" > "$F/status/$run.SMOKE"
      log "$run smoke passed"
    else
      echo "smoke failed" > "$F/status/$run.FAILED"
      log "$run smoke FAILED (see $F/logs/$run-smoke.log); point stopped"
      continue
    fi
  fi
  if [ ! -f "$F/status/$run.COLLECTED" ]; then
    if bash "$S/v2/dec/ops/m6/m6-formal.sh" 4b finalist "$point" > "$F/logs/$run-finalist.log" 2>&1; then
      echo "done" > "$F/status/$run.COLLECTED"
      log "$run collected"
    else
      echo "collection failed" > "$F/status/$run.FAILED"
      log "$run collection FAILED (see $F/logs/$run-finalist.log); point stopped"
    fi
  fi
done
printf 'track=dec-m10\nstatus=idle\npurpose=decoder M10 (formal %s finished)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$*" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "$lease"
log "formal chain finished"
