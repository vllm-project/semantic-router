#!/usr/bin/env bash
# usage: cont.sh SHA GPU NAME SEED ARM MEMBER CONTROL_RUN [--preflight]
# One Milestone 8 continuation on one GPU: v2.dec.train_dec --init decision2 from the K seed
# checkpoint MEMBER (a container path under /m4/ or /m6/) on /m8/data/m8-kd/build/ARM/train.jsonl
# (the control's TRAIN, byte for byte) with the teacher KL term (lambda = 1.0) on the rows its
# teacher.jsonl covers (--teacher-partial: rows without a teacher record train on gold only), every
# trainer setting exactly M7's control continuation (m7/cont.sh: peak learning rates 5e-6 / 5e-5,
# one checkpoint at the final update) and the trainer code of M7's mirror (TRAIN_CODE). After the
# run, lux9b.m8_rules recipe checks its provenance contract against the control run CONTROL_RUN
# (an M7 run name, e.g. C-m1) and the build's teacher row count; a FAIL stops the arm.
# --preflight first runs the zero-step and one-update runs, the recipe check of the zero-step run
# against M7's pf-C-m1-zero and v2.dec.preflight_dec (the start checkpoint as reference), which must
# PASS. Stops at the first failed step.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; seed=$4; arm=$5; member=$6; control=$7; shift 7
preflight=0
[ "${1:-}" = "--preflight" ] && preflight=1
S=$(code_dir "$sha")
J=$S/v2/9b/lux9b/m8/job.sh
host=$(host_path "$member") || { echo "member must be under /m6/ or /m4/" >&2; exit 2; }
[ -f "$host/decision_config.json" ] || { echo "no checkpoint at $member" >&2; exit 2; }
[[ "$arm" =~ ^D[12]$ ]] || { echo "ARM must be D1 or D2" >&2; exit 2; }
build=$M8/data/m8-kd/build
droot=/m8/data/m8-kd/build/$arm
[ -f "$build/manifest.json" ] && [ -s "$build/$arm/train.jsonl" ] && [ -s "$build/$arm/teacher.jsonl" ] \
  || { echo "no teacher build for $arm" >&2; exit 2; }
rows=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["arms"][sys.argv[2]]["teacher_rows"])' \
  "$build/manifest.json" "$arm")
recipe() {
  (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S:$S/v2/9b" dry python3 -m lux9b.m8_rules recipe \
    --control "$1" --arm "$2" --teacher-rows "$rows" --output "$3") > "${3%.json}.console"
}
TRAIN=(-m v2.dec.train_dec --init decision2 --model-path "$member" --train "$droot/train.jsonl"
  --select "$SELECT" --cal "$CAL" --arm "$name" --brier-weight 0.5 --weight-decay 0.01 --warmup-ratio 0.05
  --epochs 1 --batching tokens --eval-batch 2 --max-length 8192 --checkpoint-schedule every
  --save-every 1000000 --selection matrix-v1 --train-mode full --backbone-lr 5e-6 --head-lr 5e-5
  --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64
  --teacher "$droot/teacher.jsonl" --teacher-kl-weight 1.0 --teacher-partial)
if [ "$preflight" = 1 ]; then
  "$J" "$sha" "$gpu" "pf-$name-zero" "M8 preflight zero-step $name" 15 --code "$TRAIN_CODE" -- "${TRAIN[@]}" \
    --seed "$seed" --output /out/run --zero-step-only || exit 1
  recipe "$M7/pf-$control-zero/run/provenance.json" "$M8/pf-$name-zero/run/provenance.json" \
    "$M8/pf-$name-zero/recipe.json" || { echo "recipe check $name: FAIL; arm stopped" > "$M8/pf-$name-zero/STOPPED.txt"; exit 1; }
  "$J" "$sha" "$gpu" "pf-$name-one" "M8 preflight one-update $name" 20 --code "$TRAIN_CODE" -- "${TRAIN[@]}" \
    --seed "$seed" --output /out/run --max-steps 1 || exit 1
  "$J" "$sha" "$gpu" "pf-$name-check" "M8 preflight parity/reload $name" 20 --code "$TRAIN_CODE" -- -m v2.dec.preflight_dec \
    --source-path "$member" --select "$SELECT" --zero-run "/m8/pf-$name-zero/run" \
    --one-run "/m8/pf-$name-one/run" --output /out/preflight.json || exit 1
  if [ "${DRY_RUN:-0}" != 1 ]; then
    status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M8/pf-$name-check/preflight.json")
    [ "$status" = PASS ] || { echo "preflight $name: $status; arm stopped" > "$M8/pf-$name-check/STOPPED.txt"; exit 1; }
  fi
fi
"$J" "$sha" "$gpu" "$name" "M8 continuation $name seed $seed" 60 --code "$TRAIN_CODE" -- "${TRAIN[@]}" \
  --seed "$seed" --output /out/run || exit 1
[ -f "$M8/$name/run/COMPLETE.json" ] || { echo "incomplete $name" >&2; exit 1; }
recipe "$M7/$control/run/provenance.json" "$M8/$name/run/provenance.json" "$M8/$name/recipe.json" \
  || { echo "recipe check $name vs $control: FAIL; arm stopped" > "$M8/$name/STOPPED.txt"; exit 1; }
echo "done $name $(best_of "$M8/$name/run/BEST.json")"
