#!/usr/bin/env bash
# usage: cont.sh SHA GPU NAME SEED BUILD:ARM MEMBER [--preflight]
# One Milestone 7 continuation on one GPU: v2.dec.train_dec --init decision2 from the K seed
# checkpoint MEMBER (a container path under /m4/ or /m6/) on /m7/data/BUILD/build/ARM/train.jsonl
# with the own-Lux KL term on the rows its teacher.jsonl covers (--teacher-partial: rows without a
# teacher record train on gold only), every other trainer setting as the K recipe (m6/arm.sh)
# except the peak learning rates (backbone 5e-6, head 5e-5) and one checkpoint at the final update
# (--checkpoint-schedule every with a save interval beyond the horizon). --preflight first runs the
# zero-step and one-update runs and v2.dec.preflight_dec (the start checkpoint as reference), which
# must PASS. Stops at the first failed step.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; seed=$4; data=$5; member=$6; shift 6
preflight=0
[ "${1:-}" = "--preflight" ] && preflight=1
J=$(code_dir "$sha")/v2/9b/lux9b/m7/job.sh
host=$(host_path "$member") || { echo "member must be under /m6/ or /m4/" >&2; exit 2; }
[ -f "$host/decision_config.json" ] || { echo "no checkpoint at $member" >&2; exit 2; }
build=${data%%:*}; arm=${data#*:}
[[ "$data" == *:* && "$arm" =~ ^[A-Z]$ ]] || { echo "data must be BUILD:ARM" >&2; exit 2; }
droot=/m7/data/$build/build/$arm
[ -f "$M7/data/$build/build/manifest.json" ] && [ -s "$M7/data/$build/build/$arm/train.jsonl" ] \
  || { echo "no build $build arm $arm" >&2; exit 2; }
TRAIN=(-m v2.dec.train_dec --init decision2 --model-path "$member" --train "$droot/train.jsonl"
  --select "$SELECT" --cal "$CAL" --arm "$name" --brier-weight 0.5 --weight-decay 0.01 --warmup-ratio 0.05
  --epochs 1 --batching tokens --eval-batch 2 --max-length 8192 --checkpoint-schedule every
  --save-every 1000000 --selection matrix-v1 --train-mode full --backbone-lr 5e-6 --head-lr 5e-5
  --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64
  --teacher "$droot/teacher.jsonl" --teacher-kl-weight 1.0 --teacher-partial)
if [ "$preflight" = 1 ]; then
  "$J" "$sha" "$gpu" "pf-$name-zero" "M7 preflight zero-step $name" 15 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --zero-step-only || exit 1
  "$J" "$sha" "$gpu" "pf-$name-one" "M7 preflight one-update $name" 20 -- "${TRAIN[@]}" --seed "$seed" \
    --output /out/run --max-steps 1 || exit 1
  "$J" "$sha" "$gpu" "pf-$name-check" "M7 preflight parity/reload $name" 20 -- -m v2.dec.preflight_dec \
    --source-path "$member" --select "$SELECT" --zero-run "/m7/pf-$name-zero/run" \
    --one-run "/m7/pf-$name-one/run" --output /out/preflight.json || exit 1
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M7/pf-$name-check/preflight.json")
  [ "$status" = PASS ] || { echo "preflight $name: $status; arm stopped" > "$M7/pf-$name-check/STOPPED.txt"; exit 1; }
fi
"$J" "$sha" "$gpu" "$name" "M7 continuation $name seed $seed" 60 -- "${TRAIN[@]}" --seed "$seed" --output /out/run || exit 1
[ -f "$M7/$name/run/COMPLETE.json" ] || { echo "incomplete $name" >&2; exit 1; }
echo "done $name $(best_of "$M7/$name/run/BEST.json")"
