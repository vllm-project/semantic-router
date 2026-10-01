#!/usr/bin/env bash
# Decoder M15 post-training chain for one arm (prereg dec-m15-prereg-2026-10-01.md, "Development readouts and gates")
# on one M15 GPU, holding that GPU's chain flock (so it starts after the GPU's training chain and readouts never share a
# GPU):
#   1. wait until the arm's two seeds have a terminal marker;
#   2. the tier's owner chain (08b-RASDML, 2b-RASDML, 4b-LHA10SDML) reads the reference's MLX-DEV-M15 and the guard
#      validation point's (report only: 08b-RA-m12 = M12's 08b-RA soup, 4b-LHA10SD-m13 = M13's 4b-LHA10SD soup) and
#      compares it with the reference; the tier's other chain waits for the reference's MLX-DEV readout;
#   3. build the arm soup (m15-soup.sh), read its eight panels and its MLX-DEV-M15, compare with the reference.
# The references' eight panels were copied from M13 by m15-prep.sh. A failed step stops the chain.
#   node E GPU3: 08b-RASDML  node E GPU1: 08b-RA10SDML  node F GPU7: 4b-LHA10SDML  node F GPU6: 2b-RASDML
#   node F GPU3: 2b-RA10SDML
#
# usage: M15_NODE=e|f m15-post.sh launch|run <mirror-dir> <gpu> <ARM>
set -u
MODE=$1 SRC=$2 GPU=$3 ARM=$4
NODE=${M15_NODE:?set M15_NODE=e or f}
M=/data/dev2/runs/dec/m15
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m15
mkdir -p "$M/chains" "$M/logs"
OWNER=0 VAL="" VALCK=""
case $NODE:$GPU:$ARM in
  e:3:08b-RASDML) OWNER=1 VAL=08b-RA-m12 VALCK=$(cat /data/dev2/runs/dec/m12/soup/08b-RA/DONE) ;;
  e:1:08b-RA10SDML | f:3:2b-RA10SDML) ;;
  f:7:4b-LHA10SDML) OWNER=1 VAL=4b-LHA10SD-m13 VALCK=$(cat /data/dev2/runs/dec/m13/soup/4b-LHA10SD/DONE) ;;
  f:6:2b-RASDML) OWNER=1 ;;
  *) echo "no M15 post chain for $ARM on node $NODE GPU$GPU" >&2; exit 2 ;;
esac
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M15_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" "$ARM" \
    > "$M/logs/post-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$ARM.pid"
  echo "$(date -u +%FT%TZ) M15 post chain $ARM launched on node ${NODE^^} GPU$GPU from $SRC (pid $(cat "$M/chains/post-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) post-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
H=/data/dev2/models
t=${ARM%%-*}
case $t in
  2b) REF=2b-C0-f REFCK=$H/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0
    SOURCE=$H/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6 ;;
  08b) REF=08b-C0-e REFCK=$H/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e
    SOURCE=$H/Decision-1.0-Eos-0.8B/363c4a5e56afc115b1c78c837633956d0bbb63ab ;;
  4b) REF=4b-LH-f SOURCE=$H/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
    REFCK=/data/dev2/runs/dec/m10/soup/LH/build/LH-soup ;;
esac
PANELS="dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev"
lines() { M15_NODE=$NODE bash "$OPS/m15-lines.sh" "$@"; }
terminal() { [ -f "$ST/m15-$ARM-s$1.DONE" ] || [ -f "$ST/m15-$ARM-s$1.FAILED" ] || [ -f "$ST/m15-$ARM-s$1.STOPPED" ]; }
n=0
until terminal 1 && terminal 2; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds of $ARM"
  n=$((n + 1))
  sleep 60
done
if [ "$OWNER" = 1 ]; then
  # shellcheck disable=SC2086
  lines read "$SRC" "$GPU" "$REF" "$REFCK" "$SOURCE" $PANELS
  lines mlx "$SRC" "$GPU" "$REF" "$REFCK" "$SOURCE" || { log "reference MLX-DEV failed; the tier has no MLX-DEV guard"; exit 1; }
  if [ -n "$VAL" ]; then
    if ! { lines mlx "$SRC" "$GPU" "$VAL" "$VALCK" "$SOURCE" && lines mlxcmp "$SRC" "$VAL" "$REF"; }; then
      log "guard validation $VAL failed (report only)"
    fi
  fi
else
  n=0
  until [ -f "$M/lines/$REF/mlxdev.launch.json" ]; do
    [ $((n % 30)) = 0 ] && log "waiting for the reference's MLX-DEV readout"
    n=$((n + 1))
    [ $n -gt 600 ] && { log "the reference's MLX-DEV readout never appeared"; break; }
    sleep 60
  done
fi
M15_NODE=$NODE bash "$OPS/m15-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; no readout"; exit 1; }
CK=$(cat "$M/soup/$ARM/DONE")
# shellcheck disable=SC2086
lines read "$SRC" "$GPU" "$ARM" "$CK" "$SOURCE" $PANELS
log "readouts of $ARM finished"
if ! { lines mlx "$SRC" "$GPU" "$ARM" "$CK" "$SOURCE" && lines mlxcmp "$SRC" "$ARM" "$REF"; }; then
  log "MLX-DEV of $ARM failed"
fi
log "post $ARM finished"
