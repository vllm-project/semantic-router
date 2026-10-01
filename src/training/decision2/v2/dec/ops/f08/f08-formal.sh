#!/usr/bin/env bash
# Decoder 0.8B fast track: the one formal attempt for M12's 08b-RA (COORDINATION 2026-10-01 22:35; record
# dec-08bfast-formal-lock-2026-10-01.md). The M6 formal library on node E (M6_SMALL_NODE=E: image dbe5f32b,
# run_same_panel --isolate, copies of node B's frozen 0.8B masters in formal/m11/masters), outputs under
# /data/dev2/runs/dec/formal/f08 (prefix f08). Points come from formal/f08/select/08b-finalists.json
# (m10_formal_select.py --tier 08b on the frozen copies in formal/f08/inputs): 08b-C0 (DEV2.0-0.8B's weights, the
# formal-path parity run) first, then 08b-RA.
#   points: per point, m6-formal.sh 08b smoke <point> 8, then m6-formal.sh 08b finalist <point>; a failed step stops
#           the chain (a path failure on C0 must not spend 08b-RA's attempt); nothing is rerun.
#   mlx:    m6-formal.sh 08b mlx <run> per run, once node A has marked its report (<run>/V3-SEALED.json).
# Scoring is on node A (f08-score.sh). Node E GPU6-7 only (the fast-track lease). The CAL698 fits use the fast
# track's own decoder cache (formal/f08/triton-cache/08b-cal, a copy of M12's 0.8B read cache), not the node default.
#
# usage: f08-formal.sh launch|run <mirror-dir> <gpu> points|mlx <name> [<name> ...]
set -u
MODE=$1 SRC=$2 GPU=$3 KIND=$4
shift 4
case $GPU in 6 | 7) ;; *) echo "GPU $GPU on node E is not a fast-track GPU (6 or 7)" >&2; exit 2 ;; esac
case $KIND in points | mlx) ;; *) echo "kind must be points or mlx" >&2; exit 2 ;; esac
[ $# -ge 1 ] || { echo "name at least one point or run" >&2; exit 2; }
F=/data/dev2/runs/dec/formal/f08
S=/data/dev2/src/$SRC/src/training/decision2
mkdir -p "$F/logs" "$F/status"
TAG=$KIND-e$GPU-$1
if [ "$MODE" = launch ]; then
  mkdir "$F/logs/formal-$TAG.lock" 2> /dev/null || { echo "f08 formal $TAG already launched"; exit 0; }
  setsid nohup bash "$0" run "$SRC" "$GPU" "$KIND" "$@" > "$F/logs/formal-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$F/logs/formal-$TAG.pid"
  echo "$(date -u +%FT%TZ) f08 formal $TAG launched for $* (pid $(cat "$F/logs/formal-$TAG.pid"))" | tee -a "$F/OPERATIONS-f08.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) f08-formal-$TAG $*" | tee -a "$F/OPERATIONS-f08.log"; }
export M6_FORMAL_ROOT=$F M6_SELECT=$F/select M6_PREFIX=f08 M6_GPU=$GPU M6_EF_GPUS="6 7"
export M6_SMALL_NODE=E M6_SMALL_MASTER_DIR=/data/dev2/runs/dec/formal/m11/masters
export DEC_TRITON_CACHE=$F/triton-cache/08b-cal
[ -d "$DEC_TRITON_CACHE" ] || { log "no fast-track decoder cache $DEC_TRITON_CACHE for the CAL698 fit; stopped"; exit 1; }
grep -qx 'track=dec-08bfast' "/data/dev2/leases/gpu$GPU.lock/owner" 2> /dev/null \
  || { log "gpu$GPU.lock/owner is not the fast-track lease; stopped"; exit 1; }
# A finished runner entry (track=dec, last_job_end_utc, no dev2-dec container on the GPU) would refuse the next CAL
# fit's entry (M10 amendment 3): move it aside, as M10's / M12's wrappers do.
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
LIB=$S/v2/dec/ops/m6/m6-formal.sh
if [ "$KIND" = mlx ]; then
  for run in "$@"; do
    [ -f "$F/status/$run-mlx.FAILED" ] && { log "$run-mlx failed earlier; not rerun"; continue; }
    [ -f "$F/status/$run-mlx.COLLECTED" ] && continue
    clear_stale
    if bash "$LIB" 08b mlx "$run" > "$F/logs/$run-mlx.log" 2>&1; then
      echo "done" > "$F/status/$run-mlx.COLLECTED"
      log "$run-mlx collected"
    else
      echo "mlx collection failed" > "$F/status/$run-mlx.FAILED"
      log "$run-mlx collection FAILED (see $F/logs/$run-mlx.log)"
    fi
  done
  clear_stale
  log "mlx chain finished"
  exit 0
fi
for point in "$@"; do
  run=f08-$point
  [ -f "$F/status/$run.FAILED" ] && { log "$run failed earlier; not rerun; chain stopped"; exit 1; }
  clear_stale
  if [ ! -f "$F/status/$run.SMOKE" ]; then
    if bash "$LIB" 08b smoke "$point" 8 > "$F/logs/$run-smoke.log" 2>&1; then
      echo "done" > "$F/status/$run.SMOKE"
      log "$run smoke passed"
    else
      echo "smoke failed" > "$F/status/$run.FAILED"
      log "$run smoke FAILED (see $F/logs/$run-smoke.log); point and chain stopped"
      exit 1
    fi
  fi
  if [ ! -f "$F/status/$run.COLLECTED" ]; then
    clear_stale
    if bash "$LIB" 08b finalist "$point" > "$F/logs/$run-finalist.log" 2>&1; then
      echo "done" > "$F/status/$run.COLLECTED"
      log "$run collected"
    else
      echo "collection failed" > "$F/status/$run.FAILED"
      log "$run collection FAILED (see $F/logs/$run-finalist.log); point and chain stopped"
      exit 1
    fi
  fi
done
clear_stale
log "formal chain finished"
