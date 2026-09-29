#!/usr/bin/env bash
# usage: data.sh SHA STEP SPEC OUT_NAME [m6_data args...]
# CPU-only: one lux9b.m6_data step (split / ka / kh) for SPEC (path under v2/9b/lux9b/specs/) into
# /data/dev2/runs/9b/m6/data/OUT_NAME/build, in the pinned image, no network, 32 cores. Roots: m4
# (the x60 build), m3b / m3a2 (research & data's XL r2 ids, A0s-strict AutoJev targets, AutoJev
# production waves), hf (HF cache snapshots; the cache is also mounted at its host path for the
# two-level blob links), d10 (rights-clean data), repo (the mirror), m6 (earlier M6 outputs).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; step=$2; spec=$3; out_name=$4; shift 4
S=$(code_dir "$sha")
out=$M6/data/$out_name
[ -d "$S" ] || { echo "mirror $S missing" >&2; exit 2; }
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 66; }
mkdir -p "$out"
start=$(date -u +%FT%TZ)
echo "start $start sha $sha step $step spec $spec" > "$out/console.log"
set +e
dry docker run --rm --name "d2-9b-m6-data-$out_name" --network none --cpus 32 \
  -e PYTHONPATH=/code:/code/v2/9b -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src="$DATA/decision20-20260926/models/Decision-1.0-Lux-9B",dst=/model,readonly \
  --mount type=bind,src="$DATA/decision20-20260926",dst=/d10root,readonly \
  --mount type=bind,src="$DATA/dev2/hf-cache",dst=/hfc,readonly \
  --mount type=bind,src="$DATA/dev2/hf-cache",dst=/data/dev2/hf-cache,readonly \
  --mount type=bind,src="$DATA/dev2/runs/data/m3b",dst=/m3b,readonly \
  --mount type=bind,src="$DATA/dev2/runs/data/m3a2",dst=/m3a2,readonly \
  --mount type=bind,src="$M4",dst=/m4,readonly \
  --mount type=bind,src="$M6",dst=/m6,readonly \
  --mount type=bind,src="$out",dst=/out -w /code "$IMAGE" \
  python3 -m lux9b.m6_data "$step" --spec "/code/v2/9b/lux9b/specs/$spec" \
    --root hf=/hfc/datasets--llm-semantic-router--decision-2.0-training-data/snapshots \
    --root d10=/d10root --root repo=/code --root m3b=/m3b --root m3a2=/m3a2 --root m4=/m4 --root m6=/m6 \
    --output-dir /out/build "$@" \
  >> "$out/console.log" 2>&1
status=$?
set -e
printf '{"sha": "%s", "step": "%s", "spec": "%s", "start_utc": "%s", "end_utc": "%s", "exit": %s}\n' \
  "$sha" "$step" "$spec" "$start" "$(date -u +%FT%TZ)" "$status" > "$out/receipt.json"
tail -n 1 "$out/console.log"
exit $status
