#!/usr/bin/env bash
# usage: soup.sh SHA GPU NAME MEMBER_RUN...
# CPU: uniform FP32 soup (v2.dec.soup) of each member run's SELECT-chosen BEST full checkpoint
# into /data/dev2/runs/9b/m3/NAME-build/soup. GPU: CAL698 per-type temperatures for the soup
# (v2.dec.calibrate_ckpt) and gold-free typed DEV + CSS pilot predictions with them.
set -uo pipefail
sha=$1; gpu=$2; name=$3; shift 3
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
J=$S/v2/9b/lux9b/m3/job.sh
M3=/data/dev2/runs/9b/m3
image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
CAL=/hfc/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
build=$M3/$name-build
[ ! -e "$build" ] || { echo "$build exists" >&2; exit 66; }
mkdir -p "$build"
members=()
for run in "$@"; do
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$M3/$run/run/BEST.json")
  members+=(--member "/m3/$run/run/$best")
  echo "$run $best" >> "$build/members.txt"
done
date -u +%FT%TZ > "$build/start-utc.txt"
docker run --rm --network none --cpus 32 -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,src="$S",dst=/code,readonly --mount type=bind,src="$M3",dst=/m3,readonly \
  --mount type=bind,src="$build",dst=/out -w /code "$image" \
  python3 -m v2.dec.soup "${members[@]}" --output /out/soup > "$build/console.log" 2>&1 || exit 1
date -u +%FT%TZ > "$build/end-utc.txt"
"$J" "$sha" "$gpu" "$name-cal" "M3 CAL698 temperatures $name" 20 -- -m v2.dec.calibrate_ckpt \
  --checkpoint "/m3/$name-build/soup" --source-path /model --cal "$CAL" --output /out/calibration.json || exit 1
for panel in dev css-pilot; do
  "$J" "$sha" "$gpu" "$name-$panel" "M3 dev readout $name $panel" 20 -- -m v2.dec.infer_dec \
    --checkpoint "/m3/$name-build/soup" --source-path /model --calibration "/m3/$name-cal/calibration.json" \
    --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" \
    --model-id "decision2-9b-m3-$name" --model-revision "$name" || exit 1
done
echo "done $name"
