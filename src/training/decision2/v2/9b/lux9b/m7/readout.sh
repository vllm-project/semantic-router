#!/usr/bin/env bash
# usage: readout.sh SHA GPU NAME CHECKPOINT [--runtime RSHA] [--cal-from RUN | --cal-path PATH | --no-cal]
#                   [--panel P]... [--goldfree-panel P]... [--mlxdev]
# GPU: CAL698 per-type temperatures for one full checkpoint (v2.dec.calibrate_ckpt) and gold-free
# predictions with them at the formal 16,384-token limit (default panels typed DEV and CSS pilot,
# /panels/P.prompts.jsonl, e.g. ht-dev2 or pn1-dev; --goldfree-panel P reads /goldfree/P.prompts.jsonl,
# e.g. hs1-dev), into /data/dev2/runs/9b/m7/NAME-{cal,P...}. --mlxdev adds v2.dec.eval_rows on the
# MLX-DEV-9B rows (raw probabilities, as SELECT700 is read) into NAME-mlxdev. CHECKPOINT is a
# container path under /m7/, /m6/, /m4/ or /m3/. The inference code runs from the runtime mirror
# RSHA (default the incumbent's 3277dec9d); the wrappers from SHA. --cal-from RUN reuses
# /m7/RUN-cal/calibration.json and --cal-path an explicit container path (its checkpoint_sha256
# must be CHECKPOINT's; infer_dec refuses another); --no-cal reads at T = 1 (Noul yes / no and
# argmax answers do not depend on the temperatures).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; ckpt=$4; shift 4
rt=$RUNTIME_DEFAULT; panels=(); gpanels=(); calib=/m7/$name-cal/calibration.json; fit=1; mlx=0
while [ $# -gt 0 ]; do
  case "$1" in
    --runtime) rt=${2:?--runtime needs a SHA}; shift 2 ;;
    --panel) panels+=("${2:?--panel needs a name}"); shift 2 ;;
    --goldfree-panel) gpanels+=("${2:?--goldfree-panel needs a name}"); shift 2 ;;
    --cal-from) calib=/m7/${2:?--cal-from needs a run name}-cal/calibration.json; fit=0; shift 2 ;;
    --cal-path) calib=${2:?--cal-path needs a path}; fit=0; shift 2 ;;
    --no-cal) calib=""; fit=0; shift ;;
    --mlxdev) mlx=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ ${#panels[@]} -gt 0 ] || [ ${#gpanels[@]} -gt 0 ] || [ "$mlx" = 1 ] || panels=(dev css-pilot)
J=$(code_dir "$sha")/v2/9b/lux9b/m7/job.sh
host=$(host_path "$ckpt") || { echo "checkpoint must be under /m7/, /m6/, /m4/ or /m3/" >&2; exit 2; }
[ -f "$host/decision_config.json" ] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
[ -d "$(code_dir "$rt")" ] || { echo "runtime mirror $rt missing" >&2; exit 2; }
if [ "$fit" = 1 ]; then
  "$J" "$sha" "$gpu" "$name-cal" "M7 CAL698 temperatures $name" 20 --code "$rt" -- -m v2.dec.calibrate_ckpt \
    --checkpoint "$ckpt" --source-path /model --cal "$CAL" --output /out/calibration.json || exit 1
elif [ -n "$calib" ]; then
  chost=$(host_path "$calib") && [ -f "$chost" ] || { echo "no calibration $calib" >&2; exit 2; }
fi
CALARG=()
[ -z "$calib" ] || CALARG=(--calibration "$calib")
for panel in "${panels[@]}" "${gpanels[@]/#/goldfree:}"; do
  src=/panels/$panel.prompts.jsonl
  case "$panel" in goldfree:*) panel=${panel#goldfree:}; src=/goldfree/$panel.prompts.jsonl ;; esac
  "$J" "$sha" "$gpu" "$name-$panel" "M7 dev readout 16K $name $panel" 25 --code "$rt" -- -m v2.dec.infer_dec \
    --checkpoint "$ckpt" --source-path /model "${CALARG[@]}" \
    --input "$src" --output "/out/$panel.predictions.jsonl" --max-length 16384 \
    --model-id "decision2-9b-m7-$name" --model-revision "$name" || exit 1
done
if [ "$mlx" = 1 ]; then
  [ -s "$MLXDEV/panel.jsonl" ] || { echo "MLX-DEV-9B panel missing" >&2; exit 2; }
  "$J" "$sha" "$gpu" "$name-mlxdev" "M7 MLX-DEV-9B readout 16K $name" 20 --code "$rt" -- -m v2.dec.eval_rows \
    --checkpoint "$ckpt" --source-path /model --rows /m7/data/mlxdev/build/panel.jsonl --tag mlxdev \
    --output /out --max-length 16384 || exit 1
fi
echo "done $name"
