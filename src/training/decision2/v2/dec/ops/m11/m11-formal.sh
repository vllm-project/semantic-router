#!/usr/bin/env bash
# Decoder M11 formal collections (prereg "Formal and successor"; stage-2 prereg "Formal, successor, C1 and Index"): the
# M6 formal library on node E / F, outputs under /data/dev2/runs/dec/formal/m11 (prefix m11). 2b / 08b use
# M6_SMALL_NODE=E|F (image dbe5f32b, isolated runner, copies of node B's frozen masters in formal/m11/masters); 4b uses
# M6_4B_NODE=E|F (copies of formal/m5/cache-frozen{,-mlx}, as M10). Points come from m11/select/formal/<tier>-finalists.json
# (m10_formal_select.py --tier); a tier's C0 is collected first as the formal-path parity run against the stored bar.
# For each point in order: m6-formal.sh <tier> smoke <point> 8, then m6-formal.sh <tier> finalist <point>. mlx-diag
# follows after the node-A report (m6-formal.sh <tier> mlx <run>). A failed step stops that point (never rerun);
# scoring is on node A. Every job is a co-tenant (the runner's shared lease; >= 60 GB free VRAM): the GPU's owner
# entry stays M11's, the wrapper writes gpuN.lock/owner.dec-m11-formal.
# GPUs: node E GPU0-3 and node F GPU2-3 / 6-7 only (node F GPU4-5: amendment 1; node E GPU4-7 and node F GPU0-1 are
# never M11's).
#
# usage: M11_NODE=e|f m11-formal.sh launch|run <mirror-dir> <gpu> <tier> <point> [<point> ...]
set -u
MODE=$1 SRC=$2 GPU=$3 TIER=$4
shift 4
NODE=${M11_NODE:?set M11_NODE=e or f}
case $NODE in
  e) ALLOWED=" 0 1 2 3 " ;;
  f) ALLOWED=" 2 3 6 7 " ;;
  *) echo "unknown node $NODE" >&2; exit 2 ;;
esac
[[ $ALLOWED == *" $GPU "* ]] || { echo "GPU $GPU on node $NODE is not an M11 GPU" >&2; exit 2; }
case $TIER in 2b | 08b | 4b) ;; *) echo "tier must be 2b, 08b or 4b" >&2; exit 2 ;; esac
M=/data/dev2/runs/dec/m11
F=/data/dev2/runs/dec/formal/m11
S=/data/dev2/src/$SRC/src/training/decision2
mkdir -p "$F/logs" "$F/status"
TAG=$TIER-$NODE$GPU
if [ "$MODE" = launch ]; then
  mkdir "$F/logs/formal-$TAG.lock" 2> /dev/null || { echo "M11 formal $TAG already launched"; exit 0; }
  M11_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$GPU" "$TIER" "$@" > "$F/logs/formal-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$F/logs/formal-$TAG.pid"
  echo "$(date -u +%FT%TZ) M11 formal $TAG launched for $* (pid $(cat "$F/logs/formal-$TAG.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) formal-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
export M6_FORMAL_ROOT=$F M6_SELECT=$M/select/formal M6_PREFIX=m11 M6_GPU=$GPU M6_EF_GPUS="${ALLOWED# }"
if [ "$TIER" = 4b ]; then
  export M6_4B_NODE=${NODE^^}
else
  export M6_SMALL_NODE=${NODE^^} M6_SMALL_MASTER_DIR=$F/masters
fi
T=/data/dev2/runs/dec/triton-cache/dbe5f32b2263
[ -d "$T" ] || { log "no decoder cache $T for the CAL698 fit; stopped"; exit 1; }
entry=/data/dev2/leases/gpu$GPU.lock/owner.dec-m11-formal
printf 'track=dec-m11\nstatus=busy (co-tenant)\npurpose=decoder M11 formal %s %s (runner entries owner.dec-formal)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$TIER" "$*" "$(date -u +%FT%TZ)" "$(date -u -d '+120 min' +%FT%TZ)" > "$entry"
# A finished runner entry (track=dec, last_job_end_utc, no dev2-dec container on the GPU) would refuse the next CAL
# fit's entry (M10 amendment 3): move it aside, as M8's / M10's wrappers do.
clear_stale() {
  local d=/data/dev2/leases/gpu$GPU.lock e
  mkdir -p "$F/logs/stale-leases"
  for e in owner.dec-formal owner.m6-formal-smoke; do
    [ -f "$d/$e" ] || continue
    if grep -qx 'track=dec' "$d/$e" && grep -q '^last_job_end_utc=' "$d/$e" \
      && ! docker ps --format '{{.Names}}' | grep -q "^dev2-dec-gpu$GPU-"; then
      mv -n "$d/$e" "$F/logs/stale-leases/gpu$GPU-$e.$(date -u +%Y%m%dT%H%M%S.%NZ)"
      log "moved the finished decoder lease entry gpu$GPU.lock/$e to logs/stale-leases"
    fi
  done
}
for point in "$@"; do
  run=m11-$point
  [ -f "$F/status/$run.FAILED" ] && { log "$run failed earlier; not rerun"; continue; }
  clear_stale
  if [ ! -f "$F/status/$run.SMOKE" ]; then
    if bash "$S/v2/dec/ops/m6/m6-formal.sh" "$TIER" smoke "$point" 8 > "$F/logs/$run-smoke.log" 2>&1; then
      echo "done" > "$F/status/$run.SMOKE"
      log "$run smoke passed"
    else
      echo "smoke failed" > "$F/status/$run.FAILED"
      log "$run smoke FAILED (see $F/logs/$run-smoke.log); point stopped"
      continue
    fi
  fi
  if [ ! -f "$F/status/$run.COLLECTED" ]; then
    clear_stale
    if bash "$S/v2/dec/ops/m6/m6-formal.sh" "$TIER" finalist "$point" > "$F/logs/$run-finalist.log" 2>&1; then
      echo "done" > "$F/status/$run.COLLECTED"
      log "$run collected"
    else
      echo "collection failed" > "$F/status/$run.FAILED"
      log "$run collection FAILED (see $F/logs/$run-finalist.log); point stopped"
    fi
  fi
done
rm -f "$entry"
log "formal chain finished"
