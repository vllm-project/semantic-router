#!/usr/bin/env bash
# Decoder M5 soup pipeline for one arm (prereg dec-m5-prereg-2026-09-29.md; the M4 procedure), called by a
# chain after each seed: returns at once unless all three seeds' postruns are complete and no soup was
# started (mkdir lock), so the chain that finishes the arm's last seed builds it on its own GPU. Uniform
# FP32 soup of the BEST checkpoints, CAL698 temperatures, typed DEV + CSS pilot + SELECT readouts, the
# development readout against Nox 1.0 and the N4XF soup, and the MLX-DEV readout once the panel is READY
# (otherwise left to the baseline step, which reads pending soups).
# usage: m5-soup.sh <mirror-dir> <arm e.g. N5N> <gpu 3|4>
set -u
SRC=$1 G=$2 GPU=$3
M=/data/dev2/runs/dec/m5
A=$M/arms O=$M/soup/$G
S=/data/dev2/src/$SRC/src/training/decision2
L=$S/v2/dec/launch.sh
NOX=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
mkdir -p "$M/soup"
log() { echo "$(date -u +%FT%TZ) $G $*" >> "$M/soup/OPERATIONS.log"; }
for s in s1 s2 s3; do
  if grep -qE "m5-$G-$s (zero-step FAILED|one-step FAILED|gate job FAILED|full run FAILED|preflight FAIL)" "$A/OPERATIONS.log" 2>/dev/null; then
    [ -d "$O" ] || { mkdir -p "$O"; log "seed $G-$s failed; soup not built (median-seed rule needs all seeds)"; }
    exit 0
  fi
  grep -q "m5-$G-$s postrun complete" "$A/OPERATIONS.log" 2>/dev/null || exit 0
done
mkdir -p "$O"
mkdir "$O/started" 2>/dev/null || exit 0
declare -A RENDER=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU"
members=()
: > "$O/members.txt"
for s in s1 s2 s3; do
  b=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint'])" "$A/full/m5-$G-$s/BEST.json")
  members+=(--member "/runs/m5/arms/full/m5-$G-$s/$b")
  echo "$G-$s $b" >> "$O/members.txt"
done
bash "$L" "m5-$G-soup-build" "$SRC" "$O/build" --cpu -- -m v2.dec.soup "${members[@]}" --output "/out/$G-soup" \
  || { log "soup build FAILED"; exit 1; }
log "soup built on GPU$GPU: $(tail -1 "$O/build.stdout.log")"
CK=/runs/m5/soup/$G/build/$G-soup
bash "$L" "m5-$G-soup-cal698" "$SRC" "$O/cal698" -- -m v2.dec.calibrate_ckpt --checkpoint "$CK" --cal /data/cal.jsonl \
  --output /out/calibration.json || { log "cal FAILED"; exit 1; }
for panel in dev css-pilot; do
  bash "$L" "m5-$G-soup-$panel" "$SRC" "$O/$panel" -- -m v2.dec.infer_dec --checkpoint "$CK" --source-path "$NOX" \
    --calibration "/runs/m5/soup/$G/cal698/calibration.json" --input "/panels/$panel.prompts.jsonl" \
    --output "/out/$panel.predictions.jsonl" --model-id "decision2-dec-m5-$G-soup" --model-revision "$G-soup" \
    || { log "$panel FAILED"; exit 1; }
done
bash "$L" "m5-$G-soup-select" "$SRC" "$O/select" -- -m v2.dec.eval_rows --checkpoint "$CK" --rows /data/select.jsonl \
  --tag select --output /out || log "select FAILED"
C=/data/dev2/runs/dec/m2/controls R=/data/dev2/runs/dec/m4/soup/N4XF
args=(--arm "nox1=$C/nox1-dev/nox1.dev.predictions.jsonl,$C/nox1-css-pilot/nox1.css-pilot.predictions.jsonl"
  --arm "n4xf=$R/dev/dev.predictions.jsonl,$R/css-pilot/css-pilot.predictions.jsonl")
for s in s1 s2 s3; do
  P=$A/full/m5-$G-$s-post
  args+=(--arm "$s=$P/dev/dev.predictions.jsonl,$P/css-pilot/css-pilot.predictions.jsonl" --compare "nox1:$s")
done
args+=(--arm "soup=$O/dev/dev.predictions.jsonl,$O/css-pilot/css-pilot.predictions.jsonl" --compare nox1:soup --compare n4xf:soup)
if (cd "$S" && PYTHONPATH=$S python3 -m v2.dec.dev_readout --typed-gold /data/dev2/private/panels/gold/typed-dev.gold.jsonl \
  --css-gold /data/dev2/private/panels/gold/css-pilot.gold.jsonl "${args[@]}" --output "$O/readout.json" > "$O/readout.log" 2>&1); then
  log "readout done"
else
  log "readout FAILED"
fi
if [ -f "$M/mlxdev/READY" ]; then
  if bash "$S/v2/dec/ops/m5/m5-mlx.sh" "$SRC" "$GPU" "m5-$G-soup" checkpoint "$CK"; then log "mlx-dev done"; else log "mlx-dev FAILED"; fi
else
  log "mlx-dev pending (panel not READY)"
fi
