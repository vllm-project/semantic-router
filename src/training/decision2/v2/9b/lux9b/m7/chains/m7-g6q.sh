#!/usr/bin/env bash
# usage: m7-g6q.sh SHA   (started on node A by m7/launch.sh after upload_chain.sh verified it)
# 9B Milestone 7 amendment 1 chain, node A GPU6: the half-dose treatment arm Q (the continuation of
# each K seed on x60 replay + PN1-r2 once, at C's native tokens; build m7-topup-q, arm dir P). Member
# 1 with preflights, its PN1 dev readout at alpha 1, the early rule against the control's member 1
# (C-m1, already read); only if Q continues, members 2-5, then the Q5 line. Every continuation first
# checks the 24 GPU-h budget. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m7
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=6
C=m7-gpu6q
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
budget_ok 1.3 || exit 1
step Q-m1 90 -- bash "$L/cont.sh" "$SHA" "$G" Q-m1 20260931 m7-topup-q:P "${MEMBERS[0]}" --preflight || exit 1
step Q-m1-e1 10 -- bash "$L/readout.sh" "$SHA" "$G" Q-m1-e1 "/m7/Q-m1/run/$(best_of "$M7/Q-m1/run/BEST.json")" \
  --no-cal --panel pn1-dev || exit 1
bash "$L/early.sh" "$SHA" Q || exit 1
if ! early_continue Q; then
  echo "Q stopped by its early rule; no members 2-5 and no line"
  echo "chain $C done"
  exit 0
fi
for k in 2 3 4 5; do
  budget_ok 0.8 || exit 1
  step "Q-m$k" 50 -- bash "$L/cont.sh" "$SHA" "$G" "Q-m$k" "$((20260930 + k))" m7-topup-q:P "${MEMBERS[$((k - 1))]}" || exit 1
done
step Q5-line 120 -- bash "$L/line.sh" "$SHA" "$G" Q5 Q-m1 Q-m2 Q-m3 Q-m4 Q-m5 || exit 1
echo "chain $C done"
