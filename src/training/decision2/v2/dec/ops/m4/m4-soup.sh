#!/usr/bin/env bash
# Decoder M4 soup pipeline for one arm (prereg dec-m4-prereg-2026-09-29.md; the M3 procedure): wait for
# the three seeds' postrun, uniform FP32 soup of their BEST checkpoints, CAL698 temperatures, typed DEV +
# CSS pilot + SELECT readouts, then the development readout against Nox 1.0 and the M3 N4LKr soup.
# With member groups (rule 5, the cross-arm soup), the soup averages all their seeds instead.
# usage: m4-soup.sh <mirror-dir> <group e.g. N4LR> <gpu 0-4> <start path in container> [member group ...]
set -u
SRC=$1 G=$2 GPU=$3 START=$4
shift 4
MEMBERS=("$@")
[ ${#MEMBERS[@]} -eq 0 ] && MEMBERS=("$G")
M=/data/dev2/runs/dec/m4
A=$M/arms O=$M/soup/$G
mkdir -p "$O"
S=/data/dev2/src/$SRC/src/training/decision2
L=$S/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) $G $*" >> "$M/soup/OPERATIONS.log"; }
for m in "${MEMBERS[@]}"; do
  for s in s1 s2 s3; do
    until grep -q "m4-$m-$s postrun complete" "$A/OPERATIONS.log" 2>/dev/null; do
      if grep -qE "m4-$m-$s (zero-step FAILED|one-step FAILED|gate job FAILED|full run FAILED|preflight FAIL)" "$A/OPERATIONS.log" 2>/dev/null; then
        log "seed $m-$s failed; soup not built (median-seed rule needs all seeds)"
        exit 1
      fi
      sleep 60
    done
  done
done
declare -A RENDER=([0]=/dev/dri/renderD129 [1]=/dev/dri/renderD137 [2]=/dev/dri/renderD145 [3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU"
members=()
: > "$O/members.txt"
for m in "${MEMBERS[@]}"; do
  for s in s1 s2 s3; do
    b=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint'])" "$A/full/m4-$m-$s/BEST.json")
    members+=(--member "/runs/m4/arms/full/m4-$m-$s/$b")
    echo "$m-$s $b" >> "$O/members.txt"
  done
done
bash "$L" "m4-$G-soup-build" "$SRC" "$O/build" --cpu -- -m v2.dec.soup "${members[@]}" --output "/out/$G-soup" \
  || { log "soup build FAILED"; exit 1; }
log "soup built: $(tail -1 "$O/build.stdout.log")"
CK=/runs/m4/soup/$G/build/$G-soup
bash "$L" "m4-$G-soup-cal698" "$SRC" "$O/cal698" -- -m v2.dec.calibrate_ckpt --checkpoint "$CK" --cal /data/cal.jsonl \
  --output /out/calibration.json || { log "cal FAILED"; exit 1; }
for panel in dev css-pilot; do
  bash "$L" "m4-$G-soup-$panel" "$SRC" "$O/$panel" -- -m v2.dec.infer_dec --checkpoint "$CK" --source-path "$START" \
    --calibration "/runs/m4/soup/$G/cal698/calibration.json" --input "/panels/$panel.prompts.jsonl" \
    --output "/out/$panel.predictions.jsonl" --model-id "decision2-dec-m4-$G-soup" --model-revision "$G-soup" \
    || { log "$panel FAILED"; exit 1; }
done
bash "$L" "m4-$G-soup-select" "$SRC" "$O/select" -- -m v2.dec.eval_rows --checkpoint "$CK" --rows /data/select.jsonl \
  --tag select --output /out || log "select FAILED"
C=/data/dev2/runs/dec/m2/controls R=/data/dev2/runs/dec/m3/soup/N4LKr
args=(--arm "nox1=$C/nox1-dev/nox1.dev.predictions.jsonl,$C/nox1-css-pilot/nox1.css-pilot.predictions.jsonl"
  --arm "n4lkr=$R/dev/dev.predictions.jsonl,$R/css-pilot/css-pilot.predictions.jsonl")
for m in "${MEMBERS[@]}"; do
  for s in s1 s2 s3; do
    P=$A/full/m4-$m-$s-post
    name=$s
    [ ${#MEMBERS[@]} -gt 1 ] && name=$m-$s
    args+=(--arm "$name=$P/dev/dev.predictions.jsonl,$P/css-pilot/css-pilot.predictions.jsonl" --compare "nox1:$name")
  done
done
args+=(--arm "soup=$O/dev/dev.predictions.jsonl,$O/css-pilot/css-pilot.predictions.jsonl" --compare nox1:soup --compare n4lkr:soup)
if (cd "$S" && PYTHONPATH=$S python3 -m v2.dec.dev_readout --typed-gold /data/dev2/private/panels/gold/typed-dev.gold.jsonl \
  --css-gold /data/dev2/private/panels/gold/css-pilot.gold.jsonl "${args[@]}" --output "$O/readout.json" > "$O/readout.log" 2>&1); then
  log "readout done"
else
  log "readout FAILED"
fi
