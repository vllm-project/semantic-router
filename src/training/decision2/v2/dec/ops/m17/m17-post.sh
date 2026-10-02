#!/usr/bin/env bash
# Decoder M17 post-training chain for one arm (prereg dec-m17-prereg-2026-10-02.md, "Development readouts and gates") on
# one node-F GPU, holding that GPU's chain flock (so it starts after the GPU's training chain and reads never share a
# GPU):
#   1. (GPU6 only) the reference's MLX-DEV2 read: LH's own IX1 package with a fresh Triton cache, then frozen;
#   2. wait until the arm's two seeds have a terminal marker;
#   3. build the arm soup (m17-soup.sh), read its eight panels and the old MLX-DEV (report only; compared with
#      4b-LH-f's M15 read);
#   4. its MLX-DEV2 read (restaged onto LH's package, the frozen cache; after the reference read exists).
# The reference's eight panels and old MLX-DEV read were copied from M15 by m17-prep.sh. A failed step stops the point.
#   node F GPU6: 4b-LHS10SD (and the reference read)  node F GPU2: 4b-LHS17SD
#
# usage: M17_NODE=f m17-post.sh launch|run <mirror-dir> <gpu> <ARM>
set -u
MODE=$1 SRC=$2 GPU=$3 ARM=$4
NODE=${M17_NODE:?set M17_NODE=f}
M=/data/dev2/runs/dec/m17
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m17
mkdir -p "$M/chains" "$M/logs"
OWNER=0
case $NODE:$GPU:$ARM in
  f:6:4b-LHS10SD) OWNER=1 ;;
  f:2:4b-LHS17SD) ;;
  *) echo "no M17 post chain for $ARM on node $NODE GPU$GPU" >&2; exit 2 ;;
esac
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M17_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" "$ARM" \
    > "$M/logs/post-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$ARM.pid"
  echo "$(date -u +%FT%TZ) M17 post chain $ARM launched on node ${NODE^^} GPU$GPU from $SRC (pid $(cat "$M/chains/post-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) post-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
REF=4b-LH-f
SOURCE=/data/dev2/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
lines() { M17_NODE=$NODE bash "$OPS/m17-lines.sh" "$@"; }
mlx2() { M17_NODE=$NODE bash "$OPS/m17-mlx2.sh" "$@"; }
terminal() { [ -f "$ST/m17-$ARM-s$1.DONE" ] || [ -f "$ST/m17-$ARM-s$1.FAILED" ] || [ -f "$ST/m17-$ARM-s$1.STOPPED" ]; }
if [ "$OWNER" = 1 ]; then
  mlx2 ref "$SRC" "$GPU" || log "the reference's MLX-DEV2 read failed: no point can pass gate 7"
fi
n=0
until terminal 1 && terminal 2; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds of $ARM"
  n=$((n + 1))
  sleep 60
done
M17_NODE=$NODE bash "$OPS/m17-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; no readout"; exit 1; }
CK=$(cat "$M/soup/$ARM/DONE") SHA=$(cat "$M/soup/$ARM/MODEL_SHA256")
lines read "$SRC" "$GPU" "$ARM" "$CK" "$SOURCE"
log "readouts of $ARM finished"
if ! { lines mlx "$SRC" "$GPU" "$ARM" "$CK" "$SOURCE" && lines mlxcmp "$SRC" "$ARM" "$REF"; }; then
  log "old MLX-DEV of $ARM failed (report only)"
fi
n=0
until [ -f "$M/mlx2/4b-LH-f.DONE" ] || [ -f "$M/mlx2/4b-LH-f.FAILED" ]; do
  [ $((n % 30)) = 0 ] && log "waiting for the reference's MLX-DEV2 read"
  n=$((n + 1))
  [ $n -gt 600 ] && break
  sleep 60
done
mlx2 point "$SRC" "$GPU" "$ARM" "$CK" "$SHA" || log "MLX-DEV2 of $ARM failed: the point cannot pass gate 7"
printf 'track=dec-m17\nstatus=idle\npurpose=decoder M17 (post chain of %s finished)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$ARM" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
log "post $ARM finished"
