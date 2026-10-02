#!/usr/bin/env bash
# Decoder M17 formal collections (prereg "Formal and both successor paths"): the M6 formal library on node F, outputs
# under /data/dev2/runs/dec/formal/m17 (prefix m17), 4B with M6_4B_NODE=F (copies of formal/m5/cache-frozen{,-mlx}, as
# M10 / M13). Points come from m17/select/formal/4b-finalists.json (m10_formal_select.py --tier 4b; slot 0 = the LH soup,
# the formal-path parity run m17-4b-LH). For each point in order: m6-formal.sh 4b smoke <point> 8, then m6-formal.sh 4b
# finalist <point>. mlx-diag follows after node A has sealed the point's v3 report (m17-fscore.sh formal-mark). A
# failed step stops that point (never rerun); scoring is on node A. Every job is a co-tenant (the runner's shared lease;
# >= 60 GB free VRAM): the GPU's owner entry stays M17's, the wrapper writes gpuN.lock/owner.dec-m17-formal.
# GPUs: node F GPU2-3 / 6-7 only.
#
# usage: M17_NODE=f m17-formal.sh launch|run|mlx-launch|mlx-run <mirror-dir> <gpu> <point> [<point> ...]
set -u
MODE=$1 SRC=$2 GPU=$3
shift 3
NODE=${M17_NODE:?set M17_NODE=f}
[ "$NODE" = f ] || { echo "M17 formal runs on node F" >&2; exit 2; }
ALLOWED=" 2 3 6 7 "
[[ $ALLOWED == *" $GPU "* ]] || { echo "GPU $GPU on node F is not an M17 GPU" >&2; exit 2; }
TIER=4b
M=/data/dev2/runs/dec/m17
F=/data/dev2/runs/dec/formal/m17
S=/data/dev2/src/$SRC/src/training/decision2
mkdir -p "$F/logs" "$F/status"
TAG=$TIER-$NODE$GPU
case $MODE in mlx-launch | mlx-run) TAG=mlx-$TAG ;; esac
if [ "$MODE" = launch ] || [ "$MODE" = mlx-launch ]; then
  mkdir "$F/logs/formal-$TAG.lock" 2> /dev/null || { echo "M17 formal $TAG already launched"; exit 0; }
  M17_NODE=$NODE setsid nohup bash "$0" "${MODE%launch}run" "$SRC" "$GPU" "$@" > "$F/logs/formal-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$F/logs/formal-$TAG.pid"
  echo "$(date -u +%FT%TZ) M17 formal $TAG launched for $* (pid $(cat "$F/logs/formal-$TAG.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) formal-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
export M6_FORMAL_ROOT=$F M6_SELECT=$M/select/formal M6_PREFIX=m17 M6_GPU=$GPU M6_EF_GPUS="${ALLOWED# }" M6_4B_NODE=F
T=/data/dev2/runs/dec/triton-cache/dbe5f32b2263
[ -d "$T" ] || { log "no decoder cache $T for the CAL698 fit; stopped"; exit 1; }
entry=/data/dev2/leases/gpu$GPU.lock/owner.dec-m17-formal
printf 'track=dec-m17\nstatus=busy (co-tenant)\npurpose=decoder M17 formal %s (runner entries owner.dec-formal)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$*" "$(date -u +%FT%TZ)" "$(date -u -d '+120 min' +%FT%TZ)" > "$entry"
# A finished runner entry (track=dec, last_job_end_utc, no dev2-dec container on the GPU) would refuse the next CAL
# fit's entry (M10 amendment 3): move it aside, as M13's wrapper does.
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
if [ "$MODE" = mlx-run ]; then
  for point in "$@"; do
    run=m17-$point
    [ -f "$F/status/$run.MLX" ] || [ -f "$F/status/$run.MLX-FAILED" ] && { log "$run mlx-diag already attempted"; continue; }
    [ -f "$F/$run/V3-SEALED.json" ] || { log "$run v3 not sealed by node A; mlx-diag not started"; continue; }
    clear_stale
    if bash "$S/v2/dec/ops/m6/m6-formal.sh" "$TIER" mlx "$run" > "$F/logs/$run-mlx.log" 2>&1; then
      echo "done" > "$F/status/$run.MLX"
      log "$run mlx-diag collected"
    else
      echo "mlx-diag failed" > "$F/status/$run.MLX-FAILED"
      log "$run mlx-diag FAILED (see $F/logs/$run-mlx.log); point stopped"
    fi
  done
  rm -f "$entry"
  log "mlx chain finished"
  exit 0
fi
for point in "$@"; do
  run=m17-$point
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
