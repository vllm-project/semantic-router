#!/usr/bin/env bash
# usage: arm.sh SHA GPU NAME SEED DATA_NAME [--preflight] (--kl W | --no-teacher) [--teacher-partial]
#               [--runtime RSHA] [trainer extras...]
# One Milestone 6 arm on one GPU with the Milestone 4 full fine-tuning recipe (every trainer
# hyper-parameter as m4/arm.sh): optional preflights (zero-step, one update, v2.dec.preflight_dec
# parity + bitwise reload, must PASS), the full run on /m6/data/DATA_NAME/build/train.jsonl from
# the SHA mirror, then CAL698 per-type temperatures for BEST and gold-free typed DEV + CSS pilot
# predictions at the formal 16,384-token limit from the runtime mirror RSHA (default the
# incumbent's 3277dec9d). --kl W adds the own-Lux KL term with /m6/data/DATA_NAME/build/teacher.jsonl;
# strict teacher coverage unless --teacher-partial (passed through to v2.dec.train_dec: rows
# without a teacher row train on gold only). --no-teacher trains on gold only. Stops at the
# first failed step.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; seed=$4; data=$5; shift 5
preflight=0; mode=""; kl=""; partial=0; rt=$RUNTIME_DEFAULT; extras=()
while [ $# -gt 0 ]; do
  case "$1" in
    --preflight) preflight=1; shift ;;
    --kl) [ -z "$mode" ] || { echo "give one of --kl / --no-teacher" >&2; exit 2; }
      mode=kl; kl=${2:?--kl needs a weight}; shift 2 ;;
    --no-teacher) [ -z "$mode" ] || { echo "give one of --kl / --no-teacher" >&2; exit 2; }
      mode=none; shift ;;
    --teacher-partial) partial=1; shift ;;
    --runtime) rt=${2:?--runtime needs a SHA}; shift 2 ;;
    --teacher*) echo "teacher flags are set by --kl / --teacher-partial" >&2; exit 2 ;;
    *) extras+=("$1"); shift ;;
  esac
done
[ -n "$mode" ] || { echo "give one of --kl W / --no-teacher" >&2; exit 2; }
[ "$partial" = 0 ] || [ "$mode" = kl ] || { echo "--teacher-partial needs --kl" >&2; exit 2; }
J=$(code_dir "$sha")/v2/9b/lux9b/m6/job.sh
[ -d "$(code_dir "$rt")" ] || { echo "runtime mirror $rt missing" >&2; exit 2; }
# DATA_NAME m4:NAME trains on Milestone 4's build NAME (the K control on x60, byte for byte).
case "$data" in
  m4:*) droot=/m4/data/${data#m4:}/build; hroot=$M4/data/${data#m4:}/build ;;
  *) droot=/m6/data/$data/build; hroot=$M6/data/$data/build ;;
esac
[ -f "$hroot/manifest.json" ] || { echo "no build for $data" >&2; exit 2; }
TEACH=()
[ "$mode" = kl ] && TEACH=(--teacher "$droot/teacher.jsonl" --teacher-kl-weight "$kl")
[ "$partial" = 1 ] && TEACH+=(--teacher-partial)
FULL=(--train-mode full --backbone-lr 1e-5 --head-lr 1e-4 --max-batch-tokens 32768 --max-batch-rows 64
  --update-rows 64)
TRAIN=(-m v2.dec.train_dec --model-path /model --train "$droot/train.jsonl" --select "$SELECT"
  --cal "$CAL" --arm "$name" --brier-weight 0.5 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --lora-lr 5e-5 --head-lr 2.5e-5 --residual-lr 5e-4 --weight-decay 0.01 --warmup-ratio 0.05 --epochs 1
  --batching tokens --max-batch-tokens 16384 --max-batch-rows 16 --update-rows 16 --eval-batch 2
  --max-length 8192 --checkpoint-schedule even8 --selection matrix-v1 "${FULL[@]}" "${TEACH[@]}"
  "${extras[@]}")
if [ "$preflight" = 1 ]; then
  "$J" "$sha" "$gpu" "pf-$name-zero" "M6 preflight zero-step $name" 15 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --zero-step-only || exit 1
  "$J" "$sha" "$gpu" "pf-$name-one" "M6 preflight one-update $name" 20 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --max-steps 1 || exit 1
  "$J" "$sha" "$gpu" "pf-$name-check" "M6 preflight parity/reload $name" 15 -- -m v2.dec.preflight_dec \
    --source-path /model --select "$SELECT" --zero-run "/m6/pf-$name-zero/run" \
    --one-run "/m6/pf-$name-one/run" --output /out/preflight.json || exit 1
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M6/pf-$name-check/preflight.json")
  [ "$status" = PASS ] || { echo "preflight $name: $status; arm stopped" > "$M6/pf-$name-check/STOPPED.txt"; exit 1; }
fi
"$J" "$sha" "$gpu" "$name" "M6 full arm $name seed $seed" 180 -- "${TRAIN[@]}" --seed "$seed" --output /out/run || exit 1
[ -f "$M6/$name/run/COMPLETE.json" ] || { echo "incomplete $name" >&2; exit 1; }
best=$(best_of "$M6/$name/run/BEST.json")
"$J" "$sha" "$gpu" "$name-cal" "M6 CAL698 temperatures $name" 15 --code "$rt" -- -m v2.dec.calibrate_dec \
  --run-dir "/m6/$name/run" --cal "$CAL" --source-path /model --output /out/calibration.json || exit 1
for panel in dev css-pilot; do
  "$J" "$sha" "$gpu" "$name-$panel" "M6 dev readout 16K $name $panel" 25 --code "$rt" -- -m v2.dec.infer_dec \
    --checkpoint "/m6/$name/run/$best" --source-path /model --calibration "/m6/$name-cal/calibration.json" \
    --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" --max-length 16384 \
    --model-id "decision2-9b-m6-$name" --model-revision "$best" || exit 1
done
echo "done $name $best"
