#!/usr/bin/env bash
# After a decoder-track run completes: fit CAL temperatures for its frozen BEST,
# then read typed DEV and CSS pilot once with that calibration.
#
# usage: postrun.sh <source-sha> <run-name> <source-path-in-container>
#   <run-name> is the run directory relative to /data/dev2/runs/dec (e.g. c1/full/a0).
# Outputs go to /data/dev2/runs/dec/<run-name>-post/{cal,dev,css}. Uses launch.sh,
# so DEC_IMAGE / DEC_RENDER / DEC_GPU_LABEL / DEC_DATA apply unchanged.
set -euo pipefail

sha=$1 run=$2 source=$3
root=/data/dev2/runs/dec
launch=/data/dev2/src/$sha/src/training/decision2/v2/dec/launch.sh
tag=$(echo "$run" | tr '/' '-')
[[ -f $root/$run/COMPLETE.json ]] || { echo "run not complete: $run" >&2; exit 2; }
best=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$root/$run/BEST.json")
post=$root/$run-post
mkdir -p "$post"

bash "$launch" "$tag-cal" "$sha" "$post/cal" -- -m v2.dec.calibrate_dec \
  --run-dir "/runs/$run" --cal /data/cal.jsonl --source-path "$source" \
  --output /out/calibration.json
for panel in dev css-pilot; do
  bash "$launch" "$tag-$panel" "$sha" "$post/$panel" -- -m v2.dec.infer_dec \
    --checkpoint "/runs/$run/$best" --source-path "$source" \
    --calibration "/runs/$run-post/cal/calibration.json" \
    --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" \
    --model-id "decision2-dec-${tag}" --model-revision "$best"
done
echo "postrun complete: $run BEST=$best"
