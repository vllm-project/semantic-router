#!/usr/bin/env bash
# IX1 calibration study (node side, CPU): for each package, fit per-type temperatures and the Noul
# temperature + bias on its own CAL answers (<ix1>/calib/<model>/ref.jsonl), apply both transforms to
# the merged Index predictions and dual-score them. Outputs stay private.
#
# Usage: calib.sh --src DIR --panel DIR MODEL:SIZE ...
set -euo pipefail
BASE=/data/dev2/private/eval/index021
R=$BASE/ix1
src="" panel=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --src) src="$2"; shift 2 ;;
    --panel) panel="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -f "$src/.dev2-mirror.json" && -f "$panel/panel.json" && $# -ge 1 ]] || { sed -n '2,7p' "$0" >&2; exit 2; }
S="$src/src/training/decision2"
PY="$BASE/venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES=''
umask 077
for spec in "$@"; do
  model="${spec%%:*}" size="${spec##*:}"
  cal="$R/calib/$model"
  merged="$R/runs/$model/merged/results.jsonl"
  [[ -f "$cal/ref.jsonl" && -f "$merged" ]] || { echo "$model: CAL answers or merged results missing" >&2; continue; }
  PYTHONPATH="$S" "$PY" -m v2.eval.ix1.calib fit --labels "$R/calib/labels.json" --ref "$cal/ref.jsonl" --out "$cal/fit.json"
  for mode in t tb; do
    derived="$cal/index-$mode.results.jsonl"
    rm -f "$derived"
    PYTHONPATH="$S" "$PY" -m v2.eval.ix1.calib apply --fit "$cal/fit.json" --mode "$mode" --results "$merged" --out "$derived" \
      > "$cal/apply-$mode.json"
    bash "$S/v2/eval/ix1/score.sh" --src "$src" --model "$model" --size "$size" --panel "$panel" \
      --suffix "-calib-$mode" --results "$derived"
  done
done
