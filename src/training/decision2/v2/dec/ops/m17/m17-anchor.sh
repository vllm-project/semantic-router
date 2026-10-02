#!/usr/bin/env bash
# Decoder M17 anchors (prereg "Development readouts and gates", Anchors): only when node A's arms rules
# (m17/select/4b-arms.json) say fewer than two arms pass, for the better arm named there. One point per call,
# <ARM>-a<pct> with pct 33 (alpha 1/3, two thirds LH: the 9B construction) or 67 (alpha 2/3):
#   1. lineage check of LH's soup (M10) and the arm soup (ops/m16/m16_interp.py lineage, CPU container, once per arm);
#   2. W(alpha) = (1 - alpha) LH + alpha arm, per tensor in FP32 (m16_interp.py build, CPU container);
#   3. the eight panels and the old MLX-DEV (report only) on the GPU (m17-lines.sh), then MLX-DEV2 (m17-mlx2.sh point).
# Holds the GPU's chain flock. A failed lineage check stops both anchors; a failed build or read stops that point.
#
# usage: M17_NODE=f m17-anchor.sh launch|run <mirror-dir> <gpu> <ARM> <pct>
set -u
MODE=$1 SRC=$2 GPU=$3 ARM=$4 PCT=$5
NODE=${M17_NODE:?set M17_NODE=f}
M=/data/dev2/runs/dec/m17
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m17
POINT=$ARM-a$PCT
case $ARM:$PCT in
  4b-LHS10SD:33 | 4b-LHS10SD:67 | 4b-LHS17SD:33 | 4b-LHS17SD:67) ;;
  *) echo "no M17 anchor $POINT" >&2; exit 2 ;;
esac
[[ " 2 3 6 7 " == *" $GPU "* ]] || { echo "GPU $GPU on node F is not an M17 GPU" >&2; exit 2; }
mkdir -p "$M/chains" "$M/logs" "$M/anchor"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/anchor-$POINT.lock" 2> /dev/null || { echo "anchor $POINT already launched"; exit 0; }
  M17_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" "$ARM" "$PCT" \
    > "$M/logs/anchor-$POINT.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/anchor-$POINT.pid"
  echo "$(date -u +%FT%TZ) M17 anchor $POINT launched on node F GPU$GPU from $SRC (pid $(cat "$M/chains/anchor-$POINT.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) anchor-$POINT $*" | tee -a "$M/OPERATIONS.log"; }
SOURCE=/data/dev2/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
LH=/data/dev2/runs/dec/m10/soup/LH/build/LH-soup
X=$(cat "$M/soup/$ARM/DONE" 2> /dev/null) || { log "no soup of $ARM"; exit 1; }
c() { echo "/runs/${1#/data/dev2/runs/dec/}"; }
L=$M/anchor/lineage-$ARM
if [ ! -f "$L/$ARM.json" ]; then
  [ -e "$L.launch.json" ] && { log "lineage of $ARM failed earlier"; exit 1; }
  M17_NODE=$NODE bash "$OPS/m17-launch.sh" "lineage-$ARM" "$SRC" "$L" --cpu -- v2/dec/ops/m16/m16_interp.py lineage \
    --release "$(c "$LH")" --arm "$(c "$X")" --output "/out/$ARM.json" || { log "lineage of LH and $ARM FAILED"; exit 1; }
  log "lineage of LH and $ARM: PASS"
fi
B=$M/anchor/$POINT
if [ ! -f "$B/DONE" ]; then
  [ -e "$B/build.launch.json" ] && { log "build failed earlier; not rerun"; exit 1; }
  mkdir -p "$B"
  alpha=$(python3 -c 'import sys; print({"33": 1 / 3, "67": 2 / 3}[sys.argv[1]])' "$PCT")
  M17_NODE=$NODE bash "$OPS/m17-launch.sh" "interp-$POINT" "$SRC" "$B/build" --cpu -- v2/dec/ops/m16/m16_interp.py build \
    --release "$(c "$LH")" --arm "$(c "$X")" --alpha "$alpha" --output "/out/$POINT" \
    || { log "build FAILED (see $B/build.stderr.log)"; exit 1; }
  python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
    "$B/build.stdout.log" > "$B/MODEL_SHA256" || { log "no model_sha256 in the build output"; exit 1; }
  echo "$B/build/$POINT" > "$B/DONE"
  log "built W(alpha = $alpha): $(tail -c 300 "$B/build.stdout.log" | tr '\n' ' ')"
fi
CK=$(cat "$B/DONE") SHA=$(cat "$B/MODEL_SHA256")
M17_NODE=$NODE bash "$OPS/m17-lines.sh" read "$SRC" "$GPU" "$POINT" "$CK" "$SOURCE"
if ! { M17_NODE=$NODE bash "$OPS/m17-lines.sh" mlx "$SRC" "$GPU" "$POINT" "$CK" "$SOURCE" \
  && M17_NODE=$NODE bash "$OPS/m17-lines.sh" mlxcmp "$SRC" "$POINT" 4b-LH-f; }; then
  log "old MLX-DEV of $POINT failed (report only)"
fi
M17_NODE=$NODE bash "$OPS/m17-mlx2.sh" point "$SRC" "$GPU" "$POINT" "$CK" "$SHA" || log "MLX-DEV2 of $POINT failed"
printf 'track=dec-m17\nstatus=idle\npurpose=decoder M17 (anchor %s finished)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
log "anchor $POINT finished"
