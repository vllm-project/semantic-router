#!/usr/bin/env bash
# usage: readout.sh SHA GPU NAME CHECKPOINT
# GPU: CAL698 per-type temperatures for one full checkpoint (v2.dec.calibrate_ckpt) and
# gold-free typed DEV + CSS pilot predictions with them at the formal 16,384-token limit, into
# /data/dev2/runs/9b/m4/NAME-{cal,dev,css-pilot}. CHECKPOINT is a container path under /m4/ or
# /m3/ (soups, interpolations, the Lux full checkpoint, earlier members).
set -uo pipefail
sha=$1; gpu=$2; name=$3; ckpt=$4
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
J=$S/v2/9b/lux9b/m4/job.sh
CAL=/hfc/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
case "$ckpt" in
  /m4/*) host=/data/dev2/runs/9b/m4/${ckpt#/m4/} ;;
  /m3/*) host=/data/dev2/runs/9b/m3/${ckpt#/m3/} ;;
  *) echo "checkpoint must be under /m4/ or /m3/" >&2; exit 2 ;;
esac
[ -f "$host/decision_config.json" ] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
"$J" "$sha" "$gpu" "$name-cal" "M4 CAL698 temperatures $name" 20 -- -m v2.dec.calibrate_ckpt \
  --checkpoint "$ckpt" --source-path /model --cal "$CAL" --output /out/calibration.json || exit 1
for panel in dev css-pilot; do
  "$J" "$sha" "$gpu" "$name-$panel" "M4 dev readout 16K $name $panel" 25 -- -m v2.dec.infer_dec \
    --checkpoint "$ckpt" --source-path /model --calibration "/m4/$name-cal/calibration.json" \
    --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" --max-length 16384 \
    --model-id "decision2-9b-m4-$name" --model-revision "$name" || exit 1
done
echo "done $name"
