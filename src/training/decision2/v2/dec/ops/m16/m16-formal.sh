#!/usr/bin/env bash
# Decoder M16 formal collections on node B (prereg dec-m16-prereg-2026-10-01.md, "Formal, successor items, hand-offs"):
# the M6 formal library, outputs under /data/dev2/runs/dec/formal/m16 (prefix m16). 2b / 08b use M6_SMALL_NODE=B
# (image dbe5f32b, the runner without --isolate as node B's stored bars, node B's own frozen masters copied to
# formal/m16/masters by `masters`: 2B formal/m6/cache-frozen-2b{,-mlx}, 0.8B m8s/formal/cache-frozen-08b{,-mlx});
# 4b uses the library's native node-B path (formal/m5/cache-frozen{,-mlx}). Points come from
# m16/select/formal/<tier>-finalists.json (m10_formal_select.py --tier; slot 0 is the tier reference, collected first
# as the formal-path parity run). For each point in order: m6-formal.sh <tier> smoke <point> 8, then
# m6-formal.sh <tier> finalist <point>. mlx-diag follows the node-A report (m6-formal.sh <tier> mlx <run> once
# m6-relay.sh mark has placed V3-SEALED.json). A failed step stops that point (never rerun); scoring is on node A.
# Every job is a co-tenant (the runner's shared lease; >= 60 GB free VRAM): the GPU's owner entry stays M16's, the
# wrapper writes gpuN.lock/owner.dec-m16-formal. GPUs: node B GPU3 / GPU4 only (the library's node-B render map).
# Overrides for a later track's collection on the same path: M16_FORMAL_GPUS (other node-B GPUs, exported to the
# library as M6_B_GPUS), M16_FORMAL_SELECT (another select directory), M16_FORMAL_TRACK (the co-tenant entry's track);
# M6_FORCE_T1 passes through (T = 1 staging, no CAL698 fit).
#
# mlx-launch / mlx-run (amendment 1): m6-formal.sh <tier> mlx m16-<point> for each point whose v3 report node A has
# sealed (<run>/V3-SEALED.json, m16-relay.sh formal-mark); a failed mlx-diag collection stops that point.
#
# usage: m16-formal.sh masters <mirror-dir>
#        m16-formal.sh launch|run|mlx-launch|mlx-run <mirror-dir> <gpu> <tier> <point> [<point> ...]
set -u
MODE=$1 SRC=$2
M=/data/dev2/runs/dec/m16
F=/data/dev2/runs/dec/formal/m16
R=/data/dev2/runs/dec
S=/data/dev2/src/$SRC/src/training/decision2
mkdir -p "$F/logs" "$F/status"
tm() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
if [ "$MODE" = masters ]; then
  mkdir -p "$F/masters"
  for pair in "$R/formal/m6/cache-frozen-2b:cache-frozen-2b" "$R/formal/m6/cache-frozen-2b-mlx:cache-frozen-2b-mlx" \
    "$R/m8s/formal/cache-frozen-08b:cache-frozen-08b" "$R/m8s/formal/cache-frozen-08b-mlx:cache-frozen-08b-mlx"; do
    src=${pair%%:*} dst=$F/masters/${pair##*:}
    [ "$(tm "$src" | sha256sum | cut -d' ' -f1)" = "$(sha256sum "$src.sha256" | cut -d' ' -f1)" ] \
      || { echo "frozen master $src differs from its manifest" >&2; exit 1; }
    if [ ! -d "$dst" ]; then
      cp -a "$src" "$dst.tmp" && mv "$dst.tmp" "$dst" && cp -a "$src.sha256" "$dst.sha256"
    fi
    [ "$(tm "$dst" | sha256sum | cut -d' ' -f1)" = "$(sha256sum "$dst.sha256" | cut -d' ' -f1)" ] \
      || { echo "master copy $dst differs from its manifest" >&2; exit 1; }
    echo "$(date -u +%FT%TZ) master $(basename "$dst") from ${src#"$R"/} (manifest $(sha256sum "$dst.sha256" | cut -c1-16))" \
      | tee -a "$M/OPERATIONS.log"
  done
  exit 0
fi
GPU=$3 TIER=$4
shift 4
GPUS=${M16_FORMAL_GPUS:-3 4}
[[ " $GPUS " == *" $GPU "* ]] || { echo "GPU $GPU is not an M16 formal GPU (node B GPU${GPUS// / / GPU})" >&2; exit 2; }
[ -n "${M16_FORMAL_GPUS:-}" ] && export M6_B_GPUS=$GPUS
case $TIER in 2b | 08b | 4b) ;; *) echo "tier must be 2b, 08b or 4b" >&2; exit 2 ;; esac
TAG=$TIER-b$GPU
case $MODE in mlx-launch | mlx-run) TAG=mlx-$TAG ;; esac
if [ "$MODE" = launch ] || [ "$MODE" = mlx-launch ]; then
  mkdir "$F/logs/formal-$TAG.lock" 2> /dev/null || { echo "M16 formal $TAG already launched"; exit 0; }
  setsid nohup bash "$0" "${MODE%launch}run" "$SRC" "$GPU" "$TIER" "$@" > "$F/logs/formal-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$F/logs/formal-$TAG.pid"
  echo "$(date -u +%FT%TZ) M16 formal $TAG launched for $* (pid $(cat "$F/logs/formal-$TAG.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || [ "$MODE" = mlx-run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) formal-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
export M6_FORMAL_ROOT=$F M6_SELECT=${M16_FORMAL_SELECT:-$M/select/formal} M6_PREFIX=m16 M6_GPU=$GPU
if [ "$TIER" != 4b ]; then
  export M6_SMALL_NODE=B M6_SMALL_MASTER_DIR=$F/masters
  [ -f "$F/masters/cache-frozen-$TIER.sha256" ] || { log "no node-B masters for $TIER (m16-formal.sh masters)"; exit 1; }
fi
entry=/data/dev2/leases/gpu$GPU.lock/owner.dec-m16-formal
printf 'track=%s\nstatus=busy (co-tenant)\npurpose=decoder M16 formal %s %s (runner entries owner.dec-formal)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "${M16_FORMAL_TRACK:-dec-m16}" "$TIER" "$*" "$(date -u +%FT%TZ)" "$(date -u -d '+120 min' +%FT%TZ)" > "$entry"
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
if [ "$MODE" = mlx-run ]; then
  for point in "$@"; do
    run=m16-$point
    { [ -f "$F/status/$run.MLX" ] || [ -f "$F/status/$run.MLX-FAILED" ]; } && { log "$run mlx-diag already attempted"; continue; }
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
  run=m16-$point
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
