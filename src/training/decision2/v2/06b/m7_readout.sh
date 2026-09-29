#!/usr/bin/env bash
# M7 development readout of one causal 0.6B export through training.model.infer, exactly as
# m1_arm.sh's qwen-causal readout (image Python, run_container.sh, --max-length 8192, pinned
# scorers), but into a new directory and with an optional Score bias. m1_arm.sh is unchanged.
# Usage (node A): m7_readout.sh <gpu> <mirror-dir-name> <arm> <readout-name> [package|<container path>]
#   A-D3:  m7_readout.sh 0 <sha>-src_training_decision2 m7a-mxcx-sb ad3 package
#   (b):   m7_readout.sh 0 <sha>-src_training_decision2 m7-mxcx-soup dev
# <arm> is /data/dev2/runs/06b/m1/arms/<arm>/full/{best-export,COMPLETE.json}. "package" means the
# export's own score_bias.json. Output: /data/dev2/runs/06b/m7/readouts/<arm>/<readout-name>/
# (must not exist). M7_PANELS (default "dev") may add css-pilot.
set -euo pipefail
gpu="$1"
sha="$2"
arm="$3"
label="$4"
bias="${5:-}"
run="/data/dev2/src/$sha/src/training/decision2/v2/06b/run_container.sh"
[[ "$label" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "bad readout name: $label" >&2; exit 2; }
host_arm="/data/dev2/runs/06b/m1/arms/$arm"
out="/runs/m1/arms/$arm"
host_dir="/data/dev2/runs/06b/m7/readouts/$arm/$label"
dir="/runs/m7/readouts/$arm/$label"
[[ -f "$host_arm/full/COMPLETE.json" ]] || { echo "no COMPLETE.json for $arm" >&2; exit 1; }
[[ ! -e "$host_dir" ]] || { echo "$host_dir exists" >&2; exit 1; }
manifest=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_export_manifest_sha256'])" "$host_arm/full/COMPLETE.json")
[[ "$(sha256sum "$host_arm/full/best-export.MANIFEST.json" | cut -d' ' -f1)" == "$manifest" ]] || {
  echo "$arm export manifest differs from COMPLETE.json" >&2
  exit 1
}
bias_args=()
if [[ "$bias" == package ]]; then
  [[ -f "$host_arm/full/best-export/score_bias.json" ]] || { echo "$arm has no score_bias.json" >&2; exit 1; }
  bias_args=(--score-bias "$out/full/best-export/score_bias.json")
elif [[ -n "$bias" ]]; then
  bias_args=(--score-bias "$bias")
fi
mkdir -p "$host_dir"
export DEV2_PYTHON=python3
for panel in ${M7_PANELS:-dev}; do
  case "$panel" in
    dev) input=/work/runs/dev.prompts.jsonl ;;
    css-pilot) input=/work/runs/css-transfer-v1/css-pilot.prompts.jsonl ;;
    *) echo "unknown panel $panel" >&2; exit 2 ;;
  esac
  bash "$run" "$gpu" "$sha" "m7-$arm-$label-$panel" -- -m training.model.infer --checkpoint "$out/full/best-export" \
    --input "$input" --output "$dir/$panel.predictions.jsonl" --model-id "dev2-06b/$arm" \
    --model-revision "$manifest" --max-length 8192 "${bias_args[@]}"
  if [[ "$panel" == dev ]]; then
    bash "$run" "$gpu" "$sha" "m7-$arm-$label-score-dev" -- -m benchmark.score --gold /work/runs/dev.gold.jsonl \
      --predictions "$dir/dev.predictions.jsonl" --model-id "dev2-06b/$arm" --model-revision "$manifest" \
      --backend "$arm" --output "$dir/dev.score.json"
  else
    bash "$run" "$gpu" "$sha" "m7-$arm-$label-score-css" -- -m transfer.score \
      --gold /work/runs/css-transfer-v1/css-pilot.gold.jsonl \
      --predictions "$dir/css-pilot.predictions.jsonl" --output "$dir/css-pilot.score.json"
  fi
done
echo "readout done: $host_dir"
