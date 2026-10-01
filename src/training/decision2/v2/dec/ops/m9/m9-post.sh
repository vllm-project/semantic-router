#!/usr/bin/env bash
# Decoder M9 post-soup chain on node A (prereg dec-m9-prereg-2026-10-01.md + amendment 1): once the H9 soup exists
# (m9/status/H9.DONE), on GPU M9_GPU:
#   1. m9-lines.sh line H9 (alpha 1, 1/2; every panel), then score (pairs H9 - N7C) and rules -> m9/select/4b-pick.json;
#   2. only with a pick: m9-formal.sh select, smoke, finalist, score, mlx and readout (items 1-7 as information) for
#      the single formal run, labelled pilot / not a release candidate.
# Every step runs once; a failed step stops the chain (recorded, never rerun). H9.FAILED or no pick ends the chain
# without a formal run. Markers under m9/post/: <step>.DONE | <step>.FAILED, STOPPED, FINISHED.
# usage: m9-post.sh launch <mirror-dir>    (detached, under flock)
#        m9-post.sh run <mirror-dir>
set -u
MODE=$1 SRC=$2
M=/data/dev2/runs/dec/m9
P=$M/post
S=/data/dev2/src/$SRC/src/training/decision2
OPS=$S/v2/dec/ops/m9
GPU=${M9_GPU:-6}
mkdir -p "$P" "$M/logs"
if [ "$MODE" = launch ]; then
  mkdir "$P/launch.lock" 2> /dev/null || { echo "M9 post chain already launched"; exit 0; }
  M9_GPU=$GPU setsid nohup flock "$P/post.flock" bash "$0" run "$SRC" > "$M/logs/post.log" 2>&1 < /dev/null &
  echo $! > "$P/post.pid"
  echo "$(date -u +%FT%TZ) M9 post chain launched from $SRC on GPU$GPU (pid $(cat "$P/post.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
case $GPU in 6 | 7) ;; *) echo "M9_GPU must be 6 or 7" >&2; exit 2 ;; esac
export M9_GPU=$GPU
log() { echo "$(date -u +%FT%TZ) post $*" | tee -a "$M/OPERATIONS.log"; }
step() {  # <name> <command ...>: once; FAILED stops the chain
  local name=$1
  shift
  [ -f "$P/$name.DONE" ] && return 0
  [ -f "$P/$name.FAILED" ] && { log "$name failed earlier; chain stops"; exit 1; }
  log "$name start"
  if "$@" >> "$M/logs/post-$name.log" 2>&1; then
    date -u +%FT%TZ > "$P/$name.DONE"
    log "$name done"
  else
    date -u +%FT%TZ > "$P/$name.FAILED"
    log "$name FAILED (see logs/post-$name.log); chain stops"
    exit 1
  fi
}

log "post chain started (GPU$GPU, mirror $SRC); waiting for the H9 soup"
n=0
until [ -f "$M/status/H9.DONE" ]; do
  if [ -f "$M/status/H9.FAILED" ]; then
    echo "H9 has no soup: $(cat "$M/status/H9.FAILED")" > "$P/STOPPED"
    log "H9 has no soup; no lines, no formal"
    exit 0
  fi
  [ $((n % 30)) = 0 ] && log "waiting for status/H9.DONE"
  n=$((n + 1))
  sleep 60
done
step line-H9 bash "$OPS/m9-lines.sh" line H9
step score bash "$OPS/m9-lines.sh" score
step rules bash "$OPS/m9-lines.sh" rules
pick=$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["pick"]; print(p["point"] if p else "")' "$M/select/4b-pick.json")
if [ -z "$pick" ]; then
  echo "no development passer on L-H9: no formal run (prereg)" > "$P/STOPPED"
  log "no H9 pick: no formal run"
  exit 0
fi
log "H9 pick: $pick (formal run m9-$pick; pilot, not a release candidate)"
step formal-select bash "$OPS/m9-formal.sh" select
step formal-smoke bash "$OPS/m9-formal.sh" smoke "$pick"
step formal-finalist bash "$OPS/m9-formal.sh" finalist "$pick"
step formal-score bash "$OPS/m9-formal.sh" score "m9-$pick"
step formal-mlx bash "$OPS/m9-formal.sh" mlx "m9-$pick"
step formal-readout bash "$OPS/m9-formal.sh" readout "m9-$pick"
date -u +%FT%TZ > "$P/FINISHED"
log "post chain finished"
