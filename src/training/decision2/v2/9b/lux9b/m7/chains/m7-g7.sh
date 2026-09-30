#!/usr/bin/env bash
# usage: m7-g7.sh SHA   (started on node A by m7/launch.sh after upload_chain.sh verified it)
# 9B Milestone 7 chain, node A GPU7 (prereg records/lux9b-m7-prereg-2026-09-30.md): the incumbent's
# PN1 dev re-read (the early rule's reference), the matched-token control arm C (the K-mix
# continuation of each K seed): member 1 with preflights and its PN1 dev readout at alpha 1, the rest
# of the incumbent's re-read (typed DEV, CSS pilot, HT-DEV v2, MLX-DEV-9B) with its screens, members
# 2-5, the C5 line, then the report-only references K5-a12 (M6 finalist) and Lux 1.0. Every
# continuation first checks the 24 GPU-h budget. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m7
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=7
C=m7-gpu7
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
RCAL=/m6/ref-ka13-cal/calibration.json
step ref-ka13-p 10 -- bash "$L/readout.sh" "$SHA" "$G" ref-ka13 /m4/K-a13-build/soup --cal-path "$RCAL" \
  --panel pn1-dev || exit 1
budget_ok 1.3 || exit 1
step C-m1 90 -- bash "$L/cont.sh" "$SHA" "$G" C-m1 20260931 m7-topup:C "${MEMBERS[0]}" --preflight || exit 1
step C-m1-e1 10 -- bash "$L/readout.sh" "$SHA" "$G" C-m1-e1 "/m7/C-m1/run/$(best_of "$M7/C-m1/run/BEST.json")" \
  --no-cal --panel pn1-dev || exit 1
step ref-ka13-r 30 -- bash "$L/readout.sh" "$SHA" "$G" ref-ka13 /m4/K-a13-build/soup --cal-path "$RCAL" \
  --panel dev --panel css-pilot --panel ht-dev2 --mlxdev || exit 1
bash "$L/screens.sh" "$SHA" ref-ka13 ref-ka13 || exit 1
for k in 2 3 4 5; do
  budget_ok 0.8 || exit 1
  step "C-m$k" 50 -- bash "$L/cont.sh" "$SHA" "$G" "C-m$k" "$((20260930 + k))" m7-topup:C "${MEMBERS[$((k - 1))]}" || exit 1
done
step C5-line 120 -- bash "$L/line.sh" "$SHA" "$G" C5 C-m1 C-m2 C-m3 C-m4 C-m5 || exit 1
step k5a12-r 25 -- bash "$L/readout.sh" "$SHA" "$G" k5a12 /m6/K5-a12-build/soup --cal-path /m6/K5-a12-cal/calibration.json \
  --panel ht-dev2 --panel pn1-dev --mlxdev || exit 1
bash "$L/screens.sh" "$SHA" k5a12 ref-ka13 || exit 1
step lux1-r 15 -- bash "$L/readout.sh" "$SHA" "$G" lux1 /m3/pf-D-s1-zero/run/checkpoint-0000000 --no-cal \
  --panel pn1-dev --mlxdev || exit 1
bash "$L/screens.sh" "$SHA" lux1 ref-ka13 || exit 1
echo "chain $C done"
