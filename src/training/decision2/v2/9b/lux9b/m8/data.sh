#!/usr/bin/env bash
# usage: data.sh SHA STEP OUT_NAME [m8_data args...]
# CPU-only: one lux9b.m8_data step (prompts / teacher) for the spec lux9b/specs/m8-kd.json into
# /data/dev2/runs/9b/m8/data/OUT_NAME/build, in the pinned image, no network, 32 cores. Roots: m7
# (M7's runs and data: the control TRAIN and its own-Lux teacher), m3b (research & data's XL r2
# ids), m8 (earlier M8 outputs: the prompts build and the teacher predictions), repo (the mirror).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; step=$2; out_name=$3; shift 3
S=$(code_dir "$sha")
out=$M8/data/$out_name
[ -d "$S" ] || { echo "mirror $S missing" >&2; exit 2; }
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 66; }
mkdir -p "$out"
start=$(date -u +%FT%TZ)
echo "start $start sha $sha step $step" > "$out/console.log"
set +e
dry docker run --rm --name "d2-9b-m8-data-$out_name" --network none --cpus 32 \
  -e PYTHONPATH=/code:/code/v2/9b -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src="$DATA/dev2/runs/data/m3b",dst=/m3b,readonly \
  --mount type=bind,src="$M7",dst=/m7,readonly \
  --mount type=bind,src="$M8",dst=/m8,readonly \
  --mount type=bind,src="$out",dst=/out -w /code "$IMAGE" \
  python3 -m lux9b.m8_data "$step" --spec /code/v2/9b/lux9b/specs/m8-kd.json \
    --root repo=/code --root m3b=/m3b --root m7=/m7 --root m8=/m8 \
    --output-dir /out/build "$@" \
  >> "$out/console.log" 2>&1
status=$?
set -e
printf '{"sha": "%s", "step": "%s", "start_utc": "%s", "end_utc": "%s", "exit": %s}\n' \
  "$sha" "$step" "$start" "$(date -u +%FT%TZ)" "$status" > "$out/receipt.json"
tail -n 1 "$out/console.log"
exit $status
