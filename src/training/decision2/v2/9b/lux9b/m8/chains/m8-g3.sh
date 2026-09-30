#!/usr/bin/env bash
# usage: m8-g3.sh SHA   (started on node A by m8/launch.sh after upload_chain.sh verified it)
# 9B Milestone 8 chain, node A GPU3 (prereg records/lux9b-m8-prereg-2026-09-30.md + amendment 1): the
# DEV2.0-27B scored-runtime parity check (80 typed FINAL prompts; a FAIL stops the milestone), teacher
# shard 0, the M8 re-read of M7's control member C-m1 at alpha 1 (typed DEV, CSS pilot, HT-DEV v2;
# T = 1) for the early rules, the D1 / D2 teacher build once shards 1-2 (GPU4) are done (CPU), arm D1
# member 1 with preflights and its alpha-1 readouts, D1's early rule, then (if D1 continues) members
# 2-5 and the KD1 line. Every continuation first checks the 24 GPU-h budget. Stops at the first
# failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=3
C=m8-gpu3
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
step teacher-parity 20 -- bash "$L/teacher.sh" "$SHA" "$G" teacher-parity parity 80 || exit 1
step teacher-0 45 -- bash "$L/teacher.sh" "$SHA" "$G" teacher-0 shard 0 || exit 1
step C-m1-e1 25 -- bash "$L/readout.sh" "$SHA" "$G" C-m1-e1 "/m7/C-m1/run/$(best_of "$M7/C-m1/run/BEST.json")" \
  --no-cal --panel dev --panel css-pilot --panel ht-dev2 || exit 1
wait_ok teacher-1 120 || exit 1
wait_ok teacher-2 120 || exit 1
step build-kd 15 -- bash "$L/data.sh" "$SHA" teacher m8-kd --prompts-dir /m8/data/m8-prompts/build \
  --predictions 0=/m8/teacher-0/predictions.jsonl --predictions 1=/m8/teacher-1/predictions.jsonl \
  --predictions 2=/m8/teacher-2/predictions.jsonl || exit 1
budget_ok 1.3 || exit 1
step D1-m1 90 -- bash "$L/cont.sh" "$SHA" "$G" D1-m1 20260931 D1 "${MEMBERS[0]}" C-m1 --preflight || exit 1
step D1-m1-e1 20 -- bash "$L/readout.sh" "$SHA" "$G" D1-m1-e1 "/m8/D1-m1/run/$(best_of "$M8/D1-m1/run/BEST.json")" \
  --no-cal --panel dev --panel css-pilot --panel ht-dev2 --panel pn1-dev || exit 1
step early-D1 60 -- bash "$L/early.sh" "$SHA" D1 || exit 1
early_continue D1 || { echo "chain $C: D1 stopped by its early rule"; exit 0; }
for k in 2 3 4 5; do
  budget_ok 0.8 || exit 1
  step "D1-m$k" 50 -- bash "$L/cont.sh" "$SHA" "$G" "D1-m$k" "$((20260930 + k))" D1 "${MEMBERS[$((k - 1))]}" "C-m$k" || exit 1
done
step KD1-line 120 -- bash "$L/line.sh" "$SHA" "$G" KD1 D1-m1 D1-m2 D1-m3 D1-m4 D1-m5 || exit 1
echo "chain $C done"
