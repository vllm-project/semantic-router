#!/usr/bin/env bash
# usage: m6-gpu6.sh SHA   (started on node A by m6/launch.sh after upload_chain.sh verified it)
# 9B Milestone 6 chain, node A GPU6 (prereg records/lux9b-m6-prereg-2026-09-30.md): the treatment
# arms' first seeds (KA-s4, KH-s4; seed 3; preflights with each), their alpha 1/3 early-stop points
# and early rules against the control's K-s4 point (built on GPU7), then KA's second seed, soup and
# line (1/3, 1/2, 2/3) only if KA's early rule continues. Every training step first checks the
# 24 GPU-h budget. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m6
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=6
C=m6-gpu6
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
e13() { step "$1-e13" 20 -- bash "$L/interp.sh" "$SHA" "$G" "$1-e13" "/m6/$1/run/$(best_of "$M6/$1/run/BEST.json")" 1 3; }
budget_ok 7.1 || exit 1
step KA-s4 200 -- bash "$L/arm.sh" "$SHA" "$G" KA-s4 3 m6-ka --preflight --kl 1.0 || exit 1
e13 KA-s4 || exit 1
bash "$L/early.sh" "$SHA" KA-s4 K-s4 H3 || exit 1
budget_ok 7.1 || exit 1
step KH-s4 200 -- bash "$L/arm.sh" "$SHA" "$G" KH-s4 3 m6-kh --preflight --kl 1.0 --teacher-partial || exit 1
e13 KH-s4 || exit 1
bash "$L/early.sh" "$SHA" KH-s4 K-s4 RP || exit 1
if early_continue KA-s4; then
  budget_ok 6.6 || exit 1
  step KA-s5 180 -- bash "$L/arm.sh" "$SHA" "$G" KA-s5 4 m6-ka --kl 1.0 || exit 1
  step KA-soup 30 -- bash "$L/soup.sh" "$SHA" "$G" KA-soup KA-s4 KA-s5 || exit 1
  step KA-a13 20 -- bash "$L/interp.sh" "$SHA" "$G" KA-a13 /m6/KA-soup-build/soup 1 3 || exit 1
  step KA-a12 20 -- bash "$L/interp.sh" "$SHA" "$G" KA-a12 /m6/KA-soup-build/soup 1 2 || exit 1
  step KA-a23 20 -- bash "$L/interp.sh" "$SHA" "$G" KA-a23 /m6/KA-soup-build/soup 2 3 || exit 1
else
  echo "KA stopped by its early rule; no second seed or line"
fi
echo "chain $C done"
