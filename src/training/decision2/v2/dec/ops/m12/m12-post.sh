#!/usr/bin/env bash
# Decoder M12 post-training chain for one arm (prereg dec-m12-prereg-2026-10-01.md, "Development readouts") on one
# M12 GPU, holding that GPU's chain flock (so it starts after the GPU's training chain and readouts never share a GPU):
#   1. wait until the arm's two seeds have a terminal marker;
#   2. complete the tier reference's panels on this node (the M11 copies lack ib-dev): 2b-C0-e / 08b-C0-e = DEV2.0-<t>
#      weights (M8s start copies, as M11); 4b-LH-f = M10's LH soup, the weights of the released LH;
#   3. build the arm soup (m12-soup.sh) and read it.
# Panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev. A failed step stops the chain.
#   node E GPU3: 2b-RA   node E GPU1: 08b-RA   node F GPU3: 4b-LHA   node F GPU7: 4b-LHA10   (x arms: their GPU)
#
# usage: M12_NODE=e|f m12-post.sh launch|run <mirror-dir> <gpu> <ARM>
set -u
MODE=$1 SRC=$2 GPU=$3 ARM=$4
NODE=${M12_NODE:?set M12_NODE=e or f}
M=/data/dev2/runs/dec/m12
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m12
mkdir -p "$M/chains" "$M/logs"
case $NODE:$ARM in
  e:2b-RA | e:08b-RA | f:4b-LHA | f:4b-LHA10 | [ef]:4b-LHAx | [ef]:4b-LHA10x) ;;
  *) echo "no M12 post chain for $ARM on node $NODE" >&2; exit 2 ;;
esac
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M12_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" "$ARM" \
    > "$M/logs/post-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$ARM.pid"
  echo "$(date -u +%FT%TZ) M12 post chain $ARM launched on node ${NODE^^} GPU$GPU from $SRC (pid $(cat "$M/chains/post-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) post-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
H=/data/dev2/models
t=${ARM%%-*}
case $t in
  2b) REF=2b-C0-e REFCK=$H/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0
    SOURCE=$H/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6 ;;
  08b) REF=08b-C0-e REFCK=$H/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e
    SOURCE=$H/Decision-1.0-Eos-0.8B/363c4a5e56afc115b1c78c837633956d0bbb63ab ;;
  4b) REF=4b-LH-$NODE SOURCE=$H/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
    case $NODE in f) REFCK=/data/dev2/runs/dec/m10/soup/LH/build/LH-soup ;; e) REFCK=/data/dev2/runs/dec/m11/inputs/LH-soup ;; esac ;;
esac
PANELS="dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev"
read_point() {  # <point> <checkpoint>
  # shellcheck disable=SC2086
  M12_NODE=$NODE bash "$OPS/m12-lines.sh" read "$SRC" "$GPU" "$1" "$2" "$SOURCE" $PANELS
  log "readouts of $1 finished"
}
terminal() { [ -f "$ST/m12-$ARM-s$1.DONE" ] || [ -f "$ST/m12-$ARM-s$1.FAILED" ] || [ -f "$ST/m12-$ARM-s$1.STOPPED" ]; }
n=0
until terminal 1 && terminal 2; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds of $ARM"
  n=$((n + 1))
  sleep 60
done
if [ "$t" = 4b ] && [ "$NODE" = e ] && [ ! -d "$M/lines/$REF" ] && [ -d "/data/dev2/runs/dec/m11/lines/$REF" ]; then
  cp -a "/data/dev2/runs/dec/m11/lines/$REF" "$M/lines/$REF"
  log "reference readouts $REF copied from m11/lines/$REF"
fi
read_point "$REF" "$REFCK"
M12_NODE=$NODE bash "$OPS/m12-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; no readout"; exit 1; }
read_point "$ARM" "$(cat "$M/soup/$ARM/DONE")"
log "post $ARM finished"
