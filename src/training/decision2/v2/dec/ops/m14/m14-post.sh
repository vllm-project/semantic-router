#!/usr/bin/env bash
# Decoder M14 reference and post-training chains (prereg dec-m14-prereg-2026-10-01.md, "Development readouts") on one
# M14 GPU, holding that GPU's chain flock for the whole chain (readouts never share a GPU with training):
#   refs <REF> [...]  read each tier reference on this node (all eight panels): 08b-C0-a / 2b-C0-b = DEV2.0-<t>'s
#                     weights (the M12 / M13 C0 packages, copied from node E), 4b-LH-b = M10's LH soup (the weights of
#                     the released LH, copied from node F); the first read of a tier fills its read cache;
#   run <ARM>         wait until the arm's two seeds have a terminal marker, build the soup (m14-soup.sh), read it.
# Panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev. A failed step stops the chain.
# Layout: node A GPU5 refs 08b-C0-a then run 08b-RAUP; node B GPU4 refs 2b-C0-b 4b-LH-b then run 2b-RAUP; node B
# GPU3 run 4b-LHA10UP (after GPU3's training chain).
#
# usage: M14_NODE=a|b m14-post.sh launch|run|refs-launch|refs <mirror-dir> <gpu> <ARM | REF ...>
set -u
MODE=$1 SRC=$2 GPU=$3
shift 3
NODE=${M14_NODE:?set M14_NODE=a or b}
M=/data/dev2/runs/dec/m14
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m14
I=$M/inputs/refs
HF=/data/dev2/hf-cache
mkdir -p "$M/chains" "$M/logs"
source_of() {  # <tier>: the readout source path (host)
  case $1 in
    2b) echo "$HF/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6" ;;
    08b) echo "$HF/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab" ;;
    4b) echo "$HF/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b" ;;
  esac
}
ref_ck() {  # <REF>: the reference checkpoint (host)
  case $1 in
    08b-C0-a) echo "$I/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e" ;;
    2b-C0-b) echo "$I/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0" ;;
    4b-LH-b) echo "$I/LH-soup" ;;
  esac
}
case $NODE:$MODE:$* in
  a:refs*:08b-C0-a | b:refs*:2b-C0-b\ 4b-LH-b | a:launch:08b-RAUP | a:run:08b-RAUP | b:launch:2b-RAUP | b:run:2b-RAUP \
    | b:launch:4b-LHA10UP | b:run:4b-LHA10UP) ;;
  *) echo "no M14 $MODE chain for '$*' on node $NODE" >&2; exit 2 ;;
esac
case $MODE in launch) RUNMODE=run ;; refs-launch) RUNMODE=refs ;; *) RUNMODE=$MODE ;; esac
NAME=$RUNMODE-$(echo "$*" | tr ' ' '+')
if [ "$RUNMODE" != "$MODE" ]; then
  mkdir "$M/chains/post-$NAME.lock" 2> /dev/null || { echo "M14 chain $NAME already launched"; exit 0; }
  M14_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU.flock" bash "$0" "$RUNMODE" "$SRC" "$GPU" "$@" \
    > "$M/logs/post-$NAME.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$NAME.pid"
  echo "$(date -u +%FT%TZ) M14 chain $NAME launched on node ${NODE^^} GPU$GPU from $SRC (pid $(cat "$M/chains/post-$NAME.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) post-$NAME $*" | tee -a "$M/OPERATIONS.log"; }
read_point() {  # <point> <checkpoint>
  M14_NODE=$NODE bash "$OPS/m14-lines.sh" read "$SRC" "$GPU" "$1" "$2" "$(source_of "${1%%-*}")"
  log "readouts of $1 finished"
}
if [ "$MODE" = refs ]; then
  for ref in "$@"; do
    [ -d "$(ref_ck "$ref")" ] || { log "no reference checkpoint for $ref; stopped"; exit 1; }
    read_point "$ref" "$(ref_ck "$ref")"
  done
  log "references finished"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
ARM=$1
terminal() { [ -f "$ST/m14-$ARM-s$1.DONE" ] || [ -f "$ST/m14-$ARM-s$1.FAILED" ] || [ -f "$ST/m14-$ARM-s$1.STOPPED" ]; }
n=0
until terminal 1 && terminal 2; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds of $ARM"
  n=$((n + 1))
  sleep 60
done
M14_NODE=$NODE bash "$OPS/m14-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; no readout"; exit 1; }
read_point "$ARM" "$(cat "$M/soup/$ARM/DONE")"
log "post $ARM finished"
