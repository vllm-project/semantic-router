#!/usr/bin/env bash
# Decoder M17 stage-2 post chain for one arm on one node-F GPU (prereg dec-m17-stage2-prereg-2026-10-02.md,
# "Measurement and release"): a co-tenant of the GPU's training chain (no flock; the GPU's lease stays dec-m17):
#   1. wait until the arm's two seeds have a terminal marker;
#   2. its soup (m17-soup.sh: LoRA merges on this GPU, the soup on CPU);
#   3. its interpolation point 4b-<ARM>-m50 (m17-soup.sh, CPU; the prereg's arm (d));
#   4. the formal path's inputs of both: the 16K typed DEV and CSS pilot readouts (m17-lines.sh read ... dev css-pilot).
# No other development readout is run (references only). A failed step stops the chain.
#
# usage: M17_NODE=f m17-s2post.sh launch|run <mirror-dir> <gpu> <ARM>
set -u
MODE=$1 SRC=$2 GPU=$3 ARM=$4
NODE=${M17_NODE:?set M17_NODE=f}
M=/data/dev2/runs/dec/m17
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m17
mkdir -p "$M/chains" "$M/logs"
case $NODE:$GPU in f:2 | f:3 | f:6 | f:7) ;; *) echo "node $NODE GPU$GPU is not M17's" >&2; exit 2 ;; esac
case $ARM in 4b-LHS17UP | 4b-LHS23SD | 4b-LHS17IB4 | 4b-LHS17IB4X) ;; *) echo "no stage-2 arm $ARM" >&2; exit 2 ;; esac
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/s2post-$ARM.lock" 2> /dev/null || { echo "s2post $ARM already launched"; exit 0; }
  M17_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$GPU" "$ARM" > "$M/logs/s2post-$ARM.log" 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) M17 s2post $ARM launched on node ${NODE^^} GPU$GPU from $SRC (pid $!)" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) s2post-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
SOURCE=/data/dev2/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
terminal() { [ -f "$ST/m17-$ARM-s$1.DONE" ] || [ -f "$ST/m17-$ARM-s$1.FAILED" ] || [ -f "$ST/m17-$ARM-s$1.STOPPED" ]; }
n=0
until terminal 1 && terminal 2; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds"
  n=$((n + 1))
  sleep 60
done
M17_NODE=$NODE bash "$OPS/m17-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; stopped"; exit 1; }
M17_NODE=$NODE bash "$OPS/m17-soup.sh" "$SRC" "$ARM-m50" "$GPU" || log "interpolation point failed"
for p in "$ARM" "$ARM-m50"; do
  [ -f "$M/soup/$p/DONE" ] || continue
  M17_NODE=$NODE M17_COTENANT=1 bash "$OPS/m17-lines.sh" read "$SRC" "$GPU" "$p" "$(cat "$M/soup/$p/DONE")" "$SOURCE" dev css-pilot \
    || { log "formal-path readouts of $p failed"; continue; }
  log "formal-path readouts of $p finished"
done
log "finished"
