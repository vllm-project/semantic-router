#!/usr/bin/env bash
# usage: m8-g4b.sh SHA   (started on node A by m8/launch.sh after upload_chain.sh verified it)
# 9B Milestone 8 continuation chain, node A GPU4 (freed when D2 stopped by its early rule): D1
# members 3 and 5, every setting as preregistered (m8-g3b.sh runs members 2 and 4 and the KD1 line).
# Every continuation first checks the 24 GPU-h budget. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=4
C=m8-gpu4b
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
early_continue D1 || { echo "D1 did not pass its early rule" >&2; exit 1; }
for k in 3 5; do
  budget_ok 0.8 || exit 1
  step "D1-m$k" 50 -- bash "$L/cont.sh" "$SHA" "$G" "D1-m$k" "$((20260930 + k))" D1 "${MEMBERS[$((k - 1))]}" "C-m$k" || exit 1
done
echo "chain $C done"
