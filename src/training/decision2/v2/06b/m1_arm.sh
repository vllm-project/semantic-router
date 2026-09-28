#!/usr/bin/env bash
# One preregistered Milestone 1 arm on one leased GPU: preflight, full run, then the
# one-time typed DEV + CSS pilot readout of the BEST export with pinned scorers.
# Usage: m1_arm.sh <gpu-index> <commit-sha> <arm> [preflight|full|readout|all]
set -euo pipefail
gpu="$1"
sha="$2"
arm="$3"
stage="${4:-all}"
run="/data/dev2/src/$sha/src/training/decision2/v2/06b/run_container.sh"
spec="/src/src/training/decision2/v2/06b/records/arms/$arm.json"
host="/data/dev2/runs/06b/m1/arms/$arm"
out="/runs/m1/arms/$arm"
mkdir -p "$host"
spec_json="/data/dev2/src/$sha/src/training/decision2/v2/06b/records/arms/$arm.json"
field() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$1" "$2"; }

if [ "$stage" = preflight ] || [ "$stage" = all ]; then
  bash "$run" "$gpu" "$sha" "$arm-preflight" -- -m v2.06b.train --spec "$spec" --output "$out/preflight" --preflight
  status=$(field "$host/preflight/PREFLIGHT.json" "d['status']")
  echo "$arm preflight: $status"
  [ "$status" = PREFLIGHT_PASS ] || exit 3
fi
if [ "$stage" = full ] || [ "$stage" = all ]; then
  bash "$run" "$gpu" "$sha" "$arm-full" -- -m v2.06b.train --spec "$spec" --output "$out/full"
  status=$(field "$host/full/COMPLETE.json" "d['status']")
  echo "$arm full: $status"
  [ "$status" = COMPLETE ] || exit 4
fi
if [ "$stage" = readout ] || [ "$stage" = all ]; then
  manifest=$(field "$host/full/COMPLETE.json" "d['best_export_manifest_sha256']")
  family=$(field "$spec_json" "d['family']")
  if [ "$family" = kai-native ]; then
    bundle=$(field "$spec_json" "d['start']['bundle']")
    backend=$(field "$spec_json" "d['start']['backend']")
  else
    bundle=/work/models/Decision-1.0-Kai-0.6B
    backend=kai
  fi
  for panel in dev css-pilot; do
    input=/work/runs/dev.prompts.jsonl
    [ "$panel" = css-pilot ] && input=/work/runs/css-transfer-v1/css-pilot.prompts.jsonl
    bash "$run" "$gpu" "$sha" "$arm-$panel" -- -m v2.06b.predict benchmark --family "$family" --bundle "$bundle" \
      --backend "$backend" --native-dir "$out/full/best-export" --manifest-sha256 "$manifest" --input "$input" \
      --output "$out/readout/$panel.predictions.jsonl" --model-id "dev2-06b/$arm" --model-revision "$manifest" \
      --backend-label "$arm"
  done
  bash "$run" "$gpu" "$sha" "$arm-score-dev" -- -m benchmark.score --gold /work/runs/dev.gold.jsonl \
    --predictions "$out/readout/dev.predictions.jsonl" --model-id "dev2-06b/$arm" --model-revision "$manifest" \
    --backend "$arm" --output "$out/readout/dev.score.json"
  bash "$run" "$gpu" "$sha" "$arm-score-css" -- -m transfer.score --gold /work/runs/css-transfer-v1/css-pilot.gold.jsonl \
    --predictions "$out/readout/css-pilot.predictions.jsonl" --output "$out/readout/css-pilot.score.json"
  bash "$run" "$gpu" "$sha" "$arm-readout" -- -m v2.06b.readout --typed "$out/readout/dev.score.json" \
    --css "$out/readout/css-pilot.score.json" --typed-predictions "$out/readout/dev.predictions.jsonl" \
    --output "$out/readout/READOUT.json"
fi
