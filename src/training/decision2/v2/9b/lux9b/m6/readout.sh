#!/usr/bin/env bash
# usage: readout.sh SHA GPU NAME CHECKPOINT [--runtime RSHA] [--cal-from RUN] [--panel P]... [--goldfree-panel P]...
# GPU: CAL698 per-type temperatures for one full checkpoint (v2.dec.calibrate_ckpt) and gold-free
# predictions with them at the formal 16,384-token limit (default panels typed DEV and CSS pilot,
# /panels/P.prompts.jsonl; --goldfree-panel P reads /goldfree/P.prompts.jsonl, e.g. hs1-dev), into
# /data/dev2/runs/9b/m6/NAME-{cal,P...}. CHECKPOINT is a
# container path under /m6/, /m4/ or /m3/ (soups, interpolations, the incumbent K-a13, the Lux
# full checkpoint). The inference code runs from the runtime mirror RSHA (default the
# incumbent's 3277dec9d); the wrappers from SHA. --cal-from RUN reuses /m6/RUN-cal/calibration.json (its
# checkpoint_sha256 must be CHECKPOINT's; infer_dec refuses another) instead of a new CAL698 fit.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; ckpt=$4; shift 4
rt=$RUNTIME_DEFAULT; panels=(); gpanels=(); calfrom=""
while [ $# -gt 0 ]; do
  case "$1" in
    --runtime) rt=${2:?--runtime needs a SHA}; shift 2 ;;
    --panel) panels+=("${2:?--panel needs a name}"); shift 2 ;;
    --goldfree-panel) gpanels+=("${2:?--goldfree-panel needs a name}"); shift 2 ;;
    --cal-from) calfrom=${2:?--cal-from needs a run name}; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ ${#panels[@]} -gt 0 ] || [ ${#gpanels[@]} -gt 0 ] || panels=(dev css-pilot)
J=$(code_dir "$sha")/v2/9b/lux9b/m6/job.sh
host=$(host_path "$ckpt") || { echo "checkpoint must be under /m6/, /m4/ or /m3/" >&2; exit 2; }
[ -f "$host/decision_config.json" ] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
[ -d "$(code_dir "$rt")" ] || { echo "runtime mirror $rt missing" >&2; exit 2; }
calib=/m6/$name-cal/calibration.json
if [ -n "$calfrom" ]; then
  [ -f "$M6/$calfrom-cal/calibration.json" ] || { echo "no calibration $M6/$calfrom-cal" >&2; exit 2; }
  calib=/m6/$calfrom-cal/calibration.json
else
  "$J" "$sha" "$gpu" "$name-cal" "M6 CAL698 temperatures $name" 20 --code "$rt" -- -m v2.dec.calibrate_ckpt \
    --checkpoint "$ckpt" --source-path /model --cal "$CAL" --output /out/calibration.json || exit 1
fi
for panel in "${panels[@]}" "${gpanels[@]/#/goldfree:}"; do
  src=/panels/$panel.prompts.jsonl
  case "$panel" in goldfree:*) panel=${panel#goldfree:}; src=/goldfree/$panel.prompts.jsonl ;; esac
  "$J" "$sha" "$gpu" "$name-$panel" "M6 dev readout 16K $name $panel" 25 --code "$rt" -- -m v2.dec.infer_dec \
    --checkpoint "$ckpt" --source-path /model --calibration "$calib" \
    --input "$src" --output "/out/$panel.predictions.jsonl" --max-length 16384 \
    --model-id "decision2-9b-m6-$name" --model-revision "$name" || exit 1
done
echo "done $name"
