#!/usr/bin/env bash
# usage: readout.sh SHA GPU NAME CHECKPOINT [--runtime RSHA] [--panel P]...
# GPU: CAL698 per-type temperatures for one full checkpoint (v2.dec.calibrate_ckpt) and gold-free
# predictions with them at the formal 16,384-token limit (default panels typed DEV and CSS pilot,
# /panels/P.prompts.jsonl), into /data/dev2/runs/9b/m5/NAME-{cal,P...}. CHECKPOINT is a
# container path under /m5/, /m4/ or /m3/ (soups, interpolations, the incumbent K-a13, the Lux
# full checkpoint). The inference code runs from the runtime mirror RSHA (default the
# incumbent's 3277dec9d); the wrappers from SHA.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; ckpt=$4; shift 4
rt=$RUNTIME_DEFAULT; panels=()
while [ $# -gt 0 ]; do
  case "$1" in
    --runtime) rt=${2:?--runtime needs a SHA}; shift 2 ;;
    --panel) panels+=("${2:?--panel needs a name}"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ ${#panels[@]} -gt 0 ] || panels=(dev css-pilot)
J=$(code_dir "$sha")/v2/9b/lux9b/m5/job.sh
host=$(host_path "$ckpt") || { echo "checkpoint must be under /m5/, /m4/ or /m3/" >&2; exit 2; }
[ -f "$host/decision_config.json" ] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
[ -d "$(code_dir "$rt")" ] || { echo "runtime mirror $rt missing" >&2; exit 2; }
"$J" "$sha" "$gpu" "$name-cal" "M5 CAL698 temperatures $name" 20 --code "$rt" -- -m v2.dec.calibrate_ckpt \
  --checkpoint "$ckpt" --source-path /model --cal "$CAL" --output /out/calibration.json || exit 1
for panel in "${panels[@]}"; do
  "$J" "$sha" "$gpu" "$name-$panel" "M5 dev readout 16K $name $panel" 25 --code "$rt" -- -m v2.dec.infer_dec \
    --checkpoint "$ckpt" --source-path /model --calibration "/m5/$name-cal/calibration.json" \
    --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" --max-length 16384 \
    --model-id "decision2-9b-m5-$name" --model-revision "$name" || exit 1
done
echo "done $name"
