#!/usr/bin/env bash
# usage: arm.sh SHA GPU NAME SEED DATA_NAME [--preflight] (--kl W | --no-teacher) [trainer extras...]
# One Milestone 4 arm on one GPU with arm D's full fine-tuning recipe (M3 base flags + full-FT
# overrides): optional preflights (zero-step, one update, parity + bitwise reload), the full run
# on /m4/data/DATA_NAME/build/train.jsonl, CAL698 per-type temperatures for BEST, and gold-free
# typed DEV + CSS pilot predictions at the formal 16,384-token limit. --kl W adds the own-Lux KL
# term on every row with strict teacher coverage (no --teacher-partial); --no-teacher trains on
# gold only. Stops at the first failed step.
set -uo pipefail
sha=$1; gpu=$2; name=$3; seed=$4; data=$5; shift 5
preflight=0; mode=""; kl=""; extras=()
while [ $# -gt 0 ]; do
  case "$1" in
    --preflight) preflight=1; shift ;;
    --kl) [ -z "$mode" ] || { echo "give one of --kl / --no-teacher" >&2; exit 2; }
      mode=kl; kl=${2:?--kl needs a weight}; shift 2 ;;
    --no-teacher) [ -z "$mode" ] || { echo "give one of --kl / --no-teacher" >&2; exit 2; }
      mode=none; shift ;;
    --teacher*) echo "teacher flags are set by --kl" >&2; exit 2 ;;
    *) extras+=("$1"); shift ;;
  esac
done
[ -n "$mode" ] || { echo "give one of --kl W / --no-teacher" >&2; exit 2; }
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
J=$S/v2/9b/lux9b/m4/job.sh
M4=/data/dev2/runs/9b/m4
SELECT=/d10/rights_clean_goemotions_v2/select.jsonl
CAL=/hfc/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
[ -f "$M4/data/$data/build/manifest.json" ] || { echo "no build for $data" >&2; exit 2; }
TEACH=()
[ "$mode" = kl ] && TEACH=(--teacher "/m4/data/$data/build/teacher.jsonl" --teacher-kl-weight "$kl")
FULL=(--train-mode full --backbone-lr 1e-5 --head-lr 1e-4 --max-batch-tokens 32768 --max-batch-rows 64
  --update-rows 64)
TRAIN=(-m v2.dec.train_dec --model-path /model --train "/m4/data/$data/build/train.jsonl" --select "$SELECT"
  --cal "$CAL" --arm "$name" --brier-weight 0.5 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --lora-lr 5e-5 --head-lr 2.5e-5 --residual-lr 5e-4 --weight-decay 0.01 --warmup-ratio 0.05 --epochs 1
  --batching tokens --max-batch-tokens 16384 --max-batch-rows 16 --update-rows 16 --eval-batch 2
  --max-length 8192 --checkpoint-schedule even8 --selection matrix-v1 "${FULL[@]}" "${TEACH[@]}"
  "${extras[@]}")
if [ "$preflight" = 1 ]; then
  $J "$sha" "$gpu" "pf-$name-zero" "M4 preflight zero-step $name" 15 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --zero-step-only || exit 1
  $J "$sha" "$gpu" "pf-$name-one" "M4 preflight one-update $name" 20 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --max-steps 1 || exit 1
  $J "$sha" "$gpu" "pf-$name-check" "M4 preflight parity/reload $name" 15 -- -m v2.dec.preflight_dec \
    --source-path /model --select "$SELECT" --zero-run "/m4/pf-$name-zero/run" \
    --one-run "/m4/pf-$name-one/run" --output /out/preflight.json || exit 1
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M4/pf-$name-check/preflight.json")
  [ "$status" = PASS ] || { echo "preflight $name: $status; arm stopped" > "$M4/pf-$name-check/STOPPED.txt"; exit 1; }
fi
$J "$sha" "$gpu" "$name" "M4 full arm $name seed $seed" 180 -- "${TRAIN[@]}" --seed "$seed" --output /out/run || exit 1
[ -f "$M4/$name/run/COMPLETE.json" ] || { echo "incomplete $name" >&2; exit 1; }
best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$M4/$name/run/BEST.json")
$J "$sha" "$gpu" "$name-cal" "M4 CAL698 temperatures $name" 15 -- -m v2.dec.calibrate_dec \
  --run-dir "/m4/$name/run" --cal "$CAL" --source-path /model --output /out/calibration.json || exit 1
for panel in dev css-pilot; do
  $J "$sha" "$gpu" "$name-$panel" "M4 dev readout 16K $name $panel" 25 -- -m v2.dec.infer_dec \
    --checkpoint "/m4/$name/run/$best" --source-path /model --calibration "/m4/$name-cal/calibration.json" \
    --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" --max-length 16384 \
    --model-id "decision2-9b-m4-$name" --model-revision "$best" || exit 1
done
echo "done $name $best"
