#!/usr/bin/env bash
# usage: m7-g6.sh SHA   (started on node A by m7/launch.sh after upload_chain.sh verified it)
# 9B Milestone 7 chain, node A GPU6 (prereg records/lux9b-m7-prereg-2026-09-30.md): the treatment arm
# P, the PN1-r2 continuation of each K seed. Member 1 (K-s1) with preflights, its PN1 dev readout at
# alpha 1, the early rule against the matched control's member 1 (GPU7 chain); only if P continues,
# members 2-5, then the P5 line (soup, alpha 1/3 1/2 2/3 toward Lux 1.0, readouts and screens).
# Every continuation first checks the 24 GPU-h budget. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m7
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=6
C=m7-gpu6
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
budget_ok 1.3 || exit 1
step P-m1 90 -- bash "$L/cont.sh" "$SHA" "$G" P-m1 20260931 m7-topup:P "${MEMBERS[0]}" --preflight || exit 1
step P-m1-e1 10 -- bash "$L/readout.sh" "$SHA" "$G" P-m1-e1 "/m7/P-m1/run/$(best_of "$M7/P-m1/run/BEST.json")" \
  --no-cal --panel pn1-dev || exit 1
bash "$L/early.sh" "$SHA" || exit 1
if ! early_continue P; then
  echo "P stopped by its early rule; no members 2-5 and no line"
  echo "chain $C done"
  exit 0
fi
for k in 2 3 4 5; do
  budget_ok 0.8 || exit 1
  step "P-m$k" 50 -- bash "$L/cont.sh" "$SHA" "$G" "P-m$k" "$((20260930 + k))" m7-topup:P "${MEMBERS[$((k - 1))]}" || exit 1
done
step P5-line 120 -- bash "$L/line.sh" "$SHA" "$G" P5 P-m1 P-m2 P-m3 P-m4 P-m5 || exit 1
echo "chain $C done"
