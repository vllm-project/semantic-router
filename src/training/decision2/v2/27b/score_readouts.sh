#!/usr/bin/env bash
# Score ~27B development readouts and run the paired Milestone 2 contrasts (node B host, CPU).
# Usage: score_readouts.sh SRC OUT_DIR CONTROL ARM=RUN_FULL_DIR... [-- CONTRAST...]
#   Each RUN_FULL_DIR holds dev/css-pilot predictions, calibration.json and aho-* files.
#   CONTRAST: TREAT[+TREAT2]:CONTROL[+CONTROL2]:TARGET:SIGMA (see v2/27b/contrast.py).
# Scores are written once per arm (dev.score.json, css-pilot.score.json) and never overwritten.
set -euo pipefail

SRC=$1 OUT=$2 CONTROL=$3
shift 3
S=/data/dev2/src/$SRC/src/training/decision2
DEV_GOLD=/data/decision20-20260926/runs/dev.gold.jsonl
CSS_GOLD=/data/decision20-20260926/runs/css-transfer-v1/css-pilot.gold.jsonl
MX=/data/dev2/private/27b/m2-data/mixtures-v1
cd "$S"
export PYTHONPATH=$S
mkdir -p "$OUT"

arms=() contrasts=()
while [ $# -gt 0 ]; do
  [ "$1" = -- ] && { shift; contrasts=("$@"); break; }
  arms+=("$1")
  shift
done
summary_args=() contrast_args=()
for spec in "${arms[@]}"; do
  name=${spec%%=*} dir=${spec#*=}
  if [ ! -f "$dir/dev.score.json" ]; then
    python3 -m benchmark.score --gold "$DEV_GOLD" --predictions "$dir/dev.predictions.jsonl" \
      --model-id "decision2-27b-$name" --model-revision best --backend local-dynamic-candidate \
      --output "$dir/dev.score.json" > /dev/null
  fi
  if [ ! -f "$dir/css-pilot.score.json" ]; then
    python3 -m transfer.score --gold "$CSS_GOLD" --predictions "$dir/css-pilot.predictions.jsonl" \
      --output "$dir/css-pilot.score.json" > /dev/null
  fi
  summary_args+=(--arm "$name=$dir/dev.score.json,$dir/css-pilot.score.json")
  contrast_args+=(--arm "$name=$dir")
done
python3 -m v2.27b.summarize "${summary_args[@]}" --control "$CONTROL" --output "$OUT/summary.json" > /dev/null
python3 -m v2.27b.contrast --dev-gold "$DEV_GOLD" --css-gold "$CSS_GOLD" "${contrast_args[@]}" \
  --aho "A2=$MX/aho-A2.jsonl" --aho "A6g=$MX/aho-A6g.jsonl" --aho "A6h=$MX/aho-A6h.jsonl" \
  "${contrasts[@]/#/--contrast=}" --output "$OUT/contrast.json"
echo "scored ${#arms[@]} arms into $OUT"
