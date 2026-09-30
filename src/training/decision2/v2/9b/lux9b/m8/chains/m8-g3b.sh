#!/usr/bin/env bash
# usage: m8-g3b.sh SHA   (started on node A by m8/launch.sh after upload_chain.sh verified it)
# 9B Milestone 8 continuation chain, node A GPU3, after D1 passed its early rule (rules/early-D1.json;
# the first chain m8-gpu3 ended when the CPU early-rule step crashed on a screen-file race with
# D2's, and the rule was then completed on the same inputs) and D2 stopped by its early rule, which
# freed GPU4: D1 members 2 and 4 here, members 3 and 5 on GPU4 (m8-g4b.sh), every setting as
# preregistered, then the KD1 line. Every continuation first checks the 24 GPU-h budget. Stops at
# the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=3
C=m8-gpu3b
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
early_continue D1 || { echo "D1 did not pass its early rule" >&2; exit 1; }
for k in 2 4; do
  budget_ok 0.8 || exit 1
  step "D1-m$k" 50 -- bash "$L/cont.sh" "$SHA" "$G" "D1-m$k" "$((20260930 + k))" D1 "${MEMBERS[$((k - 1))]}" "C-m$k" || exit 1
done
wait_ok D1-m3 120 || exit 1
wait_ok D1-m5 120 || exit 1
for k in 3 5; do [ -f "$M8/D1-m$k/recipe.json" ] && [ ! -e "$M8/D1-m$k/STOPPED.txt" ] || { echo "D1-m$k recipe check missing or failed" >&2; exit 1; }; done
step KD1-line 120 -- bash "$L/line.sh" "$SHA" "$G" KD1 D1-m1 D1-m2 D1-m3 D1-m4 D1-m5 || exit 1
echo "chain $C done"
