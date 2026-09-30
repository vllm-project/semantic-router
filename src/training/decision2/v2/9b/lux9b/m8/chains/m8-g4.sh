#!/usr/bin/env bash
# usage: m8-g4.sh SHA   (started on node A by m8/launch.sh after upload_chain.sh verified it)
# 9B Milestone 8 chain, node A GPU4 (prereg records/lux9b-m8-prereg-2026-09-30.md + amendment 1):
# teacher shard 1, then (after GPU3's parity check passes) teacher shard 2, then (once GPU3's chain
# has built the teacher files) arm D2 member 1 with preflights and its alpha-1 readouts, D2's early
# rule, then (if D2 continues) members 2-5 and the KD2 line. Every continuation first checks the
# 24 GPU-h budget. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=4
C=m8-gpu4
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
built() { python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["exit"] == 0 else 1)' "$M8/data/m8-kd/receipt.json"; }
step teacher-1 45 -- bash "$L/teacher.sh" "$SHA" "$G" teacher-1 shard 1 || exit 1
parity_ok 60 || exit 1
step teacher-2 45 -- bash "$L/teacher.sh" "$SHA" "$G" teacher-2 shard 2 || exit 1
wait_file "$M8/data/m8-kd/receipt.json" 180 || exit 1
built || { echo "teacher build failed" >&2; exit 1; }
budget_ok 1.3 || exit 1
step D2-m1 90 -- bash "$L/cont.sh" "$SHA" "$G" D2-m1 20260931 D2 "${MEMBERS[0]}" C-m1 --preflight || exit 1
step D2-m1-e1 20 -- bash "$L/readout.sh" "$SHA" "$G" D2-m1-e1 "/m8/D2-m1/run/$(best_of "$M8/D2-m1/run/BEST.json")" \
  --no-cal --panel dev --panel css-pilot --panel ht-dev2 --panel pn1-dev || exit 1
step early-D2 60 -- bash "$L/early.sh" "$SHA" D2 || exit 1
early_continue D2 || { echo "chain $C: D2 stopped by its early rule"; exit 0; }
for k in 2 3 4 5; do
  budget_ok 0.8 || exit 1
  step "D2-m$k" 50 -- bash "$L/cont.sh" "$SHA" "$G" "D2-m$k" "$((20260930 + k))" D2 "${MEMBERS[$((k - 1))]}" "C-m$k" || exit 1
done
step KD2-line 120 -- bash "$L/line.sh" "$SHA" "$G" KD2 D2-m1 D2-m2 D2-m3 D2-m4 D2-m5 || exit 1
echo "chain $C done"
