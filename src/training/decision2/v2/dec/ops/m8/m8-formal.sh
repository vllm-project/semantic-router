#!/usr/bin/env bash
# Decoder M8 formal collections (prereg dec-m8-prereg-2026-09-30.md, "Formal (finalists only)"): the M6 formal scripts
# with M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m8, M6_SELECT=/data/dev2/runs/dec/m8/select, M6_PREFIX=m8 (2B on
# node A: M6_2B_NODE=A M6_2B_STAGE_A=1). Run from an exact mirror on the collecting node (4B node B, 2B node A).
#
#   m8-formal.sh launch|run <mirror-dir> <tier> <gpu> [<slot>[,<slot>...]]
#
# Waits for m8/select/<tier>-finalists.json (written by m8-score.sh rules on node A and relayed), then for each finalist (only the given
# slots, if any) in slot order: m6-formal.sh <tier> smoke <point> 8, then m6-formal.sh <tier> finalist <point>; on
# node A (2B) also m6-score.sh 2b <run>, m6-formal.sh 2b mlx <run> and m6-score.sh 2b mlx <run>. A failed smoke,
# calibration or collection stops that finalist (prereg stop rule, never rerun) and the next one continues. 4B runs
# are relayed and scored on node A from the workstation (m6-relay.sh pull / mark). Markers under formal/m8/status:
# <run>.SMOKE / .COLLECTED / .SCORED / .MLX (DONE files) and <run>.FAILED.
# Before each step, this track's own finished runner lease entries on the GPU (owner.dec-formal /
# owner.m6-formal-smoke with track=dec and a last_job_end_utc line, and no dev2-dec container on that GPU) move to
# formal/m8/logs/stale-leases/, so the next calibration fit is not refused by a finished run's entry (M6b incident).
set -u
MODE=$1 SRC=$2 TIER=$3 GPU=$4 SLOTS=${5:-}
F=${M8_FORMAL_ROOT:-/data/dev2/runs/dec/formal/m8}
M=${M8_ROOT:-/data/dev2/runs/dec/m8}
LEASES=${M8_LEASES:-/data/dev2/leases}
S=${M8_DECISION2:-/data/dev2/src/$SRC/src/training/decision2}
O6=$S/v2/dec/ops/m6
POLL=${M8_POLL_SECONDS:-60}
case $TIER in
  4b) [ "$GPU" = 3 ] || [ "$GPU" = 4 ] || { echo "4B collects on node B GPU3 or GPU4" >&2; exit 2; } ;;
  2b) [ "$GPU" = 5 ] || { echo "2B collects on node A GPU5" >&2; exit 2; }; export M6_2B_NODE=A M6_2B_STAGE_A=1 ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
export M6_FORMAL_ROOT=$F M6_SELECT=$M/select M6_PREFIX=m8 M6_GPU=$GPU
TAG=$TIER-gpu$GPU${SLOTS:+-slots${SLOTS//,/_}}
mkdir -p "$F/logs/stale-leases" "$F/status"
if [ "$MODE" = launch ]; then
  mkdir "$F/logs/formal-$TAG.lock" 2>/dev/null || { echo "M8 formal $TAG already launched"; exit 0; }
  setsid nohup bash "$0" run "$SRC" "$TIER" "$GPU" "$SLOTS" > "$F/logs/formal-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$F/logs/formal-$TAG.pid"
  echo "$(date -u +%FT%TZ) M8 formal $TAG launched from $SRC (pid $(cat "$F/logs/formal-$TAG.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) formal-$TAG $*" | tee -a "$M/OPERATIONS.log"; }

clear_stale() {
  local d=$LEASES/gpu$GPU.lock e
  for e in owner.dec-formal owner.m6-formal-smoke; do
    [ -f "$d/$e" ] || continue
    if grep -qx 'track=dec' "$d/$e" && grep -q '^last_job_end_utc=' "$d/$e" \
      && ! docker ps --format '{{.Names}}' | grep -q "^dev2-dec-gpu$GPU-"; then
      mv -n "$d/$e" "$F/logs/stale-leases/gpu$GPU-$e.$(date -u +%Y%m%dT%H%M%S.%NZ)"
      log "moved the finished decoder lease entry gpu$GPU.lock/$e to logs/stale-leases"
    fi
  done
}

step() {  # <run> <marker> <gpu|cpu> <description> <command ...>
  local run=$1 mark=$2 kind=$3 what=$4
  shift 4
  [ -f "$F/status/$run.$mark" ] && return 0
  clear_stale
  log "$run: $what"
  if "$@"; then
    date -u +%FT%TZ > "$F/status/$run.$mark"
    return 0
  fi
  if [ "$kind" = gpu ]; then
    echo "$what failed ($(date -u +%FT%TZ)); see $F/OPERATIONS.log" > "$F/status/$run.FAILED"
    log "$run: $what FAILED; this finalist stops (not rerun)"
  else
    echo "$what failed ($(date -u +%FT%TZ)); CPU scoring step, see $F/OPERATIONS.log" > "$F/status/$run.$mark.ERROR"
    log "$run: $what FAILED (CPU scoring step; the collection stands)"
  fi
  return 1
}

sel=$M/select/$TIER-finalists.json
n=0
until [ -f "$sel" ]; do
  [ $((n % 30)) = 0 ] && log "waiting for $sel"
  n=$((n + 1))
  sleep "$POLL"
done
points=$(python3 - "$sel" "$SLOTS" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
want = {int(s) for s in sys.argv[2].split(",") if s}
print(" ".join(f["point"] for f in sorted(doc["finalists"], key=lambda f: f["slot"]) if not want or f["slot"] in want))
EOF
) || { log "cannot read $sel"; exit 1; }
[ -n "$points" ] || { log "no finalists${SLOTS:+ in slots $SLOTS} in $sel; nothing to collect"; exit 0; }
log "finalists to collect: $points"
for point in $points; do
  run=m8-$point
  [ -f "$F/status/$run.FAILED" ] && { log "$run has a FAILED marker; skipped"; continue; }
  step "$run" SMOKE gpu "8-item smoke" bash "$O6/m6-formal.sh" "$TIER" smoke "$point" 8 || continue
  step "$run" COLLECTED gpu "formal collection" bash "$O6/m6-formal.sh" "$TIER" finalist "$point" || continue
  if [ "$TIER" = 2b ]; then
    step "$run" SCORED cpu "v3 report and comparisons" bash "$O6/m6-score.sh" 2b "$run" || continue
    step "$run" MLXCOLLECTED gpu "mlx-diag collection" bash "$O6/m6-formal.sh" 2b mlx "$run" || continue
    step "$run" MLX cpu "mlx-diag score" bash "$O6/m6-score.sh" 2b mlx "$run" || continue
  fi
  log "$run: formal steps on this node done"
done
clear_stale
log "done"
