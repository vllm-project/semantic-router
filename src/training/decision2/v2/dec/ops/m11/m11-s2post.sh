#!/usr/bin/env bash
# Decoder M11 stage-2 post chain (prereg dec-m11-stage2-prereg-2026-10-01.md, "GPUs and order") on the arm's stage-2
# GPU, co-tenant with its training chain (own lock chains/gpuN-s2post.flock):
#   1. the LH reference read on this node as 4b-LH-<node> if not read yet (node F: M10's soup m10/soup/LH/build/LH-soup;
#      node E: its hash-checked copy m11/inputs/LH-soup); on node F also its parity with M10's stored 4b-LH readouts;
#   2. once both seeds have a terminal marker: merge, soup (m11-soup.sh) and read the arm.
# Panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev. A failed step stops the chain.
# usage: M11_NODE=e|f m11-s2post.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M11_NODE:?set M11_NODE=e or f}
M=/data/dev2/runs/dec/m11
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m11
case $NODE:$GPU in
  f:7) ARM=4b-LHB LH=/data/dev2/runs/dec/m10/soup/LH/build/LH-soup ;;
  e:3) ARM=4b-LHBx LH=$M/inputs/LH-soup ;;
  *) echo "no M11 stage-2 post chain for node $NODE GPU$GPU" >&2; exit 2 ;;
esac
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M11_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU-s2post.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/post-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$ARM.pid"
  echo "$(date -u +%FT%TZ) M11 stage-2 post chain $ARM launched on node ${NODE^^} GPU$GPU from $SRC (pid $(cat "$M/chains/post-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) post-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
BASE=/data/dev2/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
PANELS="dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev"
read_point() {  # <point> <checkpoint>
  # shellcheck disable=SC2086
  M11_NODE=$NODE bash "$OPS/m11-lines.sh" read "$SRC" "$GPU" "$1" "$2" "$BASE" $PANELS
  log "readouts of $1 finished"
}

[ -f "$LH/merge_check.json" ] || { log "no LH soup at $LH; stopped"; exit 1; }
read_point "4b-LH-$NODE" "$LH"
if [ "$NODE" = f ] && [ ! -f "$M/lines/4b-LH-f/parity-4b-LH.json" ]; then
  M11_NODE=$NODE bash "$OPS/m11-lines.sh" parity "$SRC" 4b-LH-f /data/dev2/runs/dec/m10/lines/4b-LH \
    dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes
  log "4b-LH-f parity vs M10's 4b-LH: $(cat "$M/lines/4b-LH-f/parity-4b-LH.json" | tr -d '\n ' | cut -c1-300)"
fi
terminal() { [ -f "$ST/m11-$ARM-s$1.DONE" ] || [ -f "$ST/m11-$ARM-s$1.FAILED" ] || [ -f "$ST/m11-$ARM-s$1.STOPPED" ]; }
n=0
until terminal 1 && terminal 2; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds of $ARM"
  n=$((n + 1))
  sleep 60
done
M11_NODE=$NODE bash "$OPS/m11-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; no readout"; exit 1; }
read_point "$ARM" "$(cat "$M/soup/$ARM/DONE")"
log "post $ARM finished"
