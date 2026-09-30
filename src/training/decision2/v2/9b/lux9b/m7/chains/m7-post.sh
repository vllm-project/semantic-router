#!/usr/bin/env bash
# usage: m7-post.sh SHA GPU NAME   (started on node A by m7/launch.sh after NAME's lock record is pushed)
# 9B Milestone 7 formal stage for one locked finalist NAME (a line point: checkpoint
# m7/NAME-build/soup, calibration m7/NAME-cal): the formal post-key collection with the incumbent's
# runner mirror 3277dec9d (formal.sh: smoke, typed FINAL + CSS15 + public 231, mlx-diag, seal,
# report, compares, gates incl. item 7), the hs1-dev diagnostic for NAME and, once, for the incumbent
# (their CAL698 reused), its scoring (hs1.sh), and the 23:15 shipped-calibration rule (ship_cal.sh;
# derive_t1.sh when it ships T = 1). Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}; G=${2:?gpu}; NAME=${3:?finalist}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m7
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
C=m7-post-$NAME
RUNNER=3277dec9d81708fa374405ca884043443bab1b49
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
[ -f "$M7/$NAME-build/soup/decision_config.json" ] && [ -f "$M7/$NAME-cal/calibration.json" ] \
  || { echo "no checkpoint or calibration for $NAME" >&2; exit 2; }
budget_ok 1.0 || exit 1
step "$NAME-formal" 40 -- bash "$L/formal.sh" "$RUNNER" "$G" "$NAME" "$M7/$NAME-build/soup" "$M7/$NAME-cal" \
  "post-key same-panel" || exit 1
step "$NAME-h" 15 -- bash "$L/readout.sh" "$SHA" "$G" "$NAME-h" "/m7/$NAME-build/soup" --cal-from "$NAME" \
  --goldfree-panel hs1-dev || exit 1
if [ ! -f "$M7/ref-ka13-h-hs1-dev/exit-code.txt" ]; then
  step ref-ka13-h 15 -- bash "$L/readout.sh" "$SHA" "$G" ref-ka13-h /m4/K-a13-build/soup \
    --cal-path /m6/ref-ka13-cal/calibration.json --goldfree-panel hs1-dev || exit 1
fi
bash "$L/hs1.sh" "$SHA" "$NAME" "$NAME-h-hs1-dev" ref-ka13-h-hs1-dev || exit 1
ship=$(bash "$L/ship_cal.sh" "$SHA" "$NAME" "$NAME" | tail -n 1) || exit 1
echo "$ship"
if [ "$ship" = ship=T1 ]; then
  bash "$L/derive_t1.sh" "$SHA" "$NAME" || exit 1
fi
echo "chain $C done"
