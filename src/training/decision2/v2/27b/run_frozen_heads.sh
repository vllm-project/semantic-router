#!/usr/bin/env bash
# CPU-only frozen-head training or readout on cached probe features (no GPU used).
# Usage: run_frozen_heads.sh MIRROR_SHA train|readout KEY [KEY ...]
set -euo pipefail

SHA=$1 MODE=$2
shift 2
CODE=/data/dev2/src/$SHA/src/training/decision2
ROOT=/data/dev2/runs/27b/A3-probes
BENCH=/data/decision20-20260926/runs
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1

for KEY in "$@"; do
  PROBE=$ROOT/$KEY
  [ -f "$PROBE/probe.json" ] || { echo "probe receipt missing for $KEY" >&2; exit 2; }
  if [ "$MODE" = train ]; then
    ARGS=(train --features /probe/features --probe-receipt /probe/probe.json --output /probe/heads)
  elif [ "$MODE" = readout ]; then
    ARGS=(readout --features /probe/features --probe-receipt /probe/probe.json
      --selection-dir /probe/heads --dev /inputs/dev.prompts.jsonl
      --css-pilot /inputs/css-pilot.prompts.jsonl --model-id "decision2-27b-frozen-$KEY"
      --output /probe/readout)
  else
    echo "mode must be train or readout" >&2
    exit 2
  fi
  started=$(date +%s)
  docker run --rm --network none --cpus 32 -e OMP_NUM_THREADS=32 -e PYTHONPATH=/code \
    -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 \
    --mount "type=bind,src=$CODE,dst=/code,readonly" --mount "type=bind,src=$PROBE,dst=/probe" \
    --mount "type=bind,src=$BENCH/dev.prompts.jsonl,dst=/inputs/dev.prompts.jsonl,readonly" \
    --mount "type=bind,src=$BENCH/css-transfer-v1/css-pilot.prompts.jsonl,dst=/inputs/css-pilot.prompts.jsonl,readonly" \
    -w /code "$IMAGE" -m v2.27b.frozen_head "${ARGS[@]}" > "$PROBE/$MODE.log" 2>&1
  echo "{\"key\":\"$KEY\",\"mode\":\"$MODE\",\"cpu_wall_seconds\":$(($(date +%s) - started)),\"gpu_hours\":0}" | tee -a "$ROOT/cpu-jobs.jsonl"
done
