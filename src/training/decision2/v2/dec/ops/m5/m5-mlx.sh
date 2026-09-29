#!/usr/bin/env bash
# Decoder M5 MLX-DEV readout of one model (prereg dec-m5-prereg-2026-09-29.md): v2.dec.eval_rows on the
# panel (raw probabilities, as SELECT700 is read), then the MLX-DEV score and the paired group-clustered
# bootstrap against the Nox 1.0 and N4XF-soup baselines when their readouts exist. A finished readout is
# not repeated.
# usage: m5-mlx.sh <mirror-dir> <gpu 3|4> <name> package|checkpoint <model path in container>
set -u
SRC=$1 GPU=$2 NAME=$3 KIND=$4 MODEL=$5
M=/data/dev2/runs/dec/m5
X=$M/mlxdev
O=$X/readouts/$NAME
S=/data/dev2/src/$SRC/src/training/decision2
L=$S/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) mlx $NAME $*" >> "$X/OPERATIONS.log"; }
[ -f "$X/READY" ] || { log "panel not READY; skipped"; exit 1; }
declare -A RENDER=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU"
mkdir -p "$X/readouts"
if [ ! -f "$O/mlxdev-predictions.jsonl" ]; then
  if [ -f "$O.launch.json" ]; then
    log "earlier readout failed (receipt $O.launch.json); not rerun"
    exit 1
  fi
  bash "$L" "m5-mlx-$NAME" "$SRC" "$O" -- -m v2.dec.eval_rows "--$KIND" "$MODEL" --rows /runs/m5/mlxdev/build/panel.jsonl \
    --tag mlxdev --output /out || { log "readout FAILED"; exit 1; }
  log "readout done: $(tail -1 "$O.stdout.log")"
fi
I=$X/build/panel.jsonl.index.jsonl
py() { (cd "$S" && PYTHONPATH=$S python3 -m v2.dec.mlx_dev "$@"); }
[ -f "$O/score.json" ] || py score --index "$I" --predictions "$O/mlxdev-predictions.jsonl" --output "$O/score.json" >> "$X/OPERATIONS.log" 2>&1 \
  || { log "score FAILED"; exit 1; }
for ref in nox1 n4xf-soup; do
  P=$X/readouts/$ref/mlxdev-predictions.jsonl
  [ "$ref" = "$NAME" ] && continue
  [ -f "$P" ] || continue
  [ -f "$O/vs-$ref.json" ] && continue
  py compare --index "$I" --a "$P" --b "$O/mlxdev-predictions.jsonl" --output "$O/vs-$ref.json" >> "$X/OPERATIONS.log" 2>&1 \
    || log "compare vs $ref FAILED"
done
log "scored: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: round(d[k], 4) for k in ("noul_ml", "choice_ml", "score_ml", "m_dev", "noul_pred_yes_rate_macro")})' "$O/score.json")"
