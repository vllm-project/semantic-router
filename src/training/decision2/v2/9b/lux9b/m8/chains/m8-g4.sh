#!/usr/bin/env bash
# usage: m8-g4.sh SHA   (started on node A by m8/launch.sh after upload_chain.sh verified it)
# 9B Milestone 8 chain, node A GPU4 (lent by ~27B; prereg records/lux9b-m8-prereg-2026-09-30.md):
# (after GPU2's teacher parity check passes) teacher shard 2, the M8 re-read of M7's control member C-m1 at alpha 1 (typed DEV, CSS pilot,
# HT-DEV v2; T = 1) for the early rules, then member 5 of each arm that continues, and the KDX line
# (the uniform soup of all ten members) only if both arms continue. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=4
C=m8-gpu4
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
parity_ok 60 || exit 1
step teacher-2 45 -- bash "$L/teacher.sh" "$SHA" "$G" teacher-2 shard 2 || exit 1
step C-m1-e1 25 -- bash "$L/readout.sh" "$SHA" "$G" C-m1-e1 "/m7/C-m1/run/$(best_of "$M7/C-m1/run/BEST.json")" \
  --no-cal --panel dev --panel css-pilot --panel ht-dev2 || exit 1
for arm in D1 D2; do
  w=0
  until arm_decided "$arm"; do [ "$w" -lt 300 ] || exit 1; sleep 60; w=$((w + 1)); done
done
both=1
for arm in D1 D2; do
  if early_continue "$arm"; then
    budget_ok 0.8 || exit 1
    step "$arm-m5" 50 -- bash "$L/cont.sh" "$SHA" "$G" "$arm-m5" 20260935 "$arm" "${MEMBERS[4]}" C-m5 || exit 1
  else
    both=0
  fi
done
[ "$both" = 1 ] || { echo "chain $C: an arm stopped early, no KDX line"; exit 0; }
for run in D1-m2 D1-m3 D1-m4 D2-m2 D2-m3 D2-m4; do wait_ok "$run" 240 || exit 1; done
step KDX-line 120 -- bash "$L/line.sh" "$SHA" "$G" KDX D1-m1 D1-m2 D1-m3 D1-m4 D1-m5 D2-m1 D2-m2 D2-m3 D2-m4 D2-m5 || exit 1
echo "chain $C done"
