#!/usr/bin/env bash
# usage: data.sh SHA SPEC OUT_NAME
# CPU-only: materialize TRAIN + teacher file for SPEC (path under v2/9b/lux9b/specs/) into
# /data/dev2/runs/9b/m3/data/OUT_NAME with lux9b.m3_data, in the pinned image, no network.
set -euo pipefail
sha=$1; spec=$2; out_name=$3
M3=/data/dev2/runs/9b/m3
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
out=$M3/data/$out_name
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 66; }
mkdir -p "$out"
start=$(date -u +%FT%TZ)
set +e
docker run --rm --network none --cpus 48 -e PYTHONPATH=/code:/code/v2/9b -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src=/data/decision20-20260926/models/Decision-1.0-Lux-9B,dst=/model,readonly \
  --mount type=bind,src=/data/decision20-20260926,dst=/d10root,readonly \
  --mount type=bind,src=/data/dev2/hf-cache,dst=/hfc,readonly \
  --mount type=bind,src="$out",dst=/out -w /code "$image" \
  python3 -m lux9b.m3_data --spec "/code/v2/9b/lux9b/specs/$spec" \
    --root hf=/hfc/datasets--llm-semantic-router--decision-2.0-training-data/snapshots \
    --root d10=/d10root --root repo=/code --tokenizer /model --workers 48 --output-dir /out/build \
  > "$out/console.log" 2>&1
status=$?
set -e
printf '{"sha": "%s", "spec": "%s", "start_utc": "%s", "end_utc": "%s", "exit": %s}\n' \
  "$sha" "$spec" "$start" "$(date -u +%FT%TZ)" "$status" > "$out/receipt.json"
exit $status
