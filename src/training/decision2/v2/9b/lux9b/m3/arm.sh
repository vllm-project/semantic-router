#!/usr/bin/env bash
# usage: arm.sh SHA GPU NAME SEED DATA_NAME [--preflight] [--no-teacher] [trainer extras...]
# One Milestone 3 arm on one GPU: optional preflights (zero-step, one update, parity +
# bitwise reload), the full run, CAL698 per-type temperatures for BEST, and gold-free typed
# DEV + CSS pilot predictions with those temperatures. Stops at the first failed step.
set -uo pipefail
sha=$1; gpu=$2; name=$3; seed=$4; data=$5; shift 5
preflight=0; teacher=1; extras=()
for a in "$@"; do
  case "$a" in
    --preflight) preflight=1 ;;
    --no-teacher) teacher=0 ;;
    *) extras+=("$a") ;;
  esac
done
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
J=$S/v2/9b/lux9b/m3/job.sh
M3=/data/dev2/runs/9b/m3
SELECT=/d10/rights_clean_goemotions_v2/select.jsonl
CAL=/hfds/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
TEACH=()
[ "$teacher" = 1 ] && TEACH=(--teacher "/m3/data/$data/build/teacher.jsonl" --teacher-kl-weight 0.5 --teacher-partial)
TRAIN=(-m v2.dec.train_dec --model-path /model --train "/m3/data/$data/build/train.jsonl" --select "$SELECT"
  --cal "$CAL" --arm "$name" --brier-weight 0.5 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --lora-lr 5e-5 --head-lr 2.5e-5 --residual-lr 5e-4 --weight-decay 0.01 --warmup-ratio 0.05 --epochs 1
  --batching tokens --max-batch-tokens 16384 --max-batch-rows 16 --update-rows 16 --eval-batch 2
  --max-length 8192 --checkpoint-schedule even8 --selection matrix-v1 "${TEACH[@]}" "${extras[@]}")
if [ "$preflight" = 1 ]; then
  $J "$sha" "$gpu" "pf-$name-zero" "M3 preflight zero-step $name" 15 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --zero-step-only || exit 1
  $J "$sha" "$gpu" "pf-$name-one" "M3 preflight one-update $name" 20 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --max-steps 1 || exit 1
  $J "$sha" "$gpu" "pf-$name-check" "M3 preflight parity/reload $name" 15 -- -m v2.dec.preflight_dec \
    --source-path /model --select "$SELECT" --zero-run "/m3/pf-$name-zero/run" \
    --one-run "/m3/pf-$name-one/run" --output /out/preflight.json || exit 1
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M3/pf-$name-check/preflight.json")
  [ "$status" = PASS ] || { echo "preflight $name: $status; arm stopped" > "$M3/pf-$name-check/STOPPED.txt"; exit 1; }
fi
$J "$sha" "$gpu" "$name" "M3 full arm $name seed $seed" 120 -- "${TRAIN[@]}" --seed "$seed" --output /out/run || exit 1
[ -f "$M3/$name/run/COMPLETE.json" ] || { echo "incomplete $name" >&2; exit 1; }
best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$M3/$name/run/BEST.json")
$J "$sha" "$gpu" "$name-cal" "M3 CAL698 temperatures $name" 15 -- -m v2.dec.calibrate_dec \
  --run-dir "/m3/$name/run" --cal "$CAL" --source-path /model --output /out/calibration.json || exit 1
for panel in dev css-pilot; do
  $J "$sha" "$gpu" "$name-$panel" "M3 dev readout $name $panel" 20 -- -m v2.dec.infer_dec \
    --checkpoint "/m3/$name/run/$best" --source-path /model --calibration "/m3/$name-cal/calibration.json" \
    --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" \
    --model-id "decision2-9b-m3-$name" --model-revision "$best" || exit 1
done
echo "done $name $best"
