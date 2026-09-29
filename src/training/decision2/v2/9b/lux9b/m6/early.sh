#!/usr/bin/env bash
# usage: early.sh SHA ARM CONTROL PROTECT
# CPU, node A: the preregistered early stop of arm ARM at its first full checkpoint. Waits (up to
# 6 h) for the readouts of ARM-e13 and CONTROL-e13 (interp.sh: 1/3 of each first seed's BEST + 2/3
# Lux 1.0), scores both (score.sh early-ARM), and applies lux9b.m6_rules early (P gain >= 0.5 and
# the protected screen PROTECT, H3 or RP, not below the control's) into
# /data/dev2/runs/9b/m6/rules/early-ARM.json. Prints the decision; exits 0 whether it continues
# or stops (early_continue ARM reads the file), non-zero only on a failed step.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; arm=$2; ctl=$3; protect=$4
OWN=$(code_dir "$sha")
L=$OWN/v2/9b/lux9b/m6
for run in "$arm-e13-css-pilot" "$ctl-e13-css-pilot" "$arm-e13-dev" "$ctl-e13-dev"; do
  wait_ok "$run" 360 || exit 1
done
if [ ! -f "$M6/early-$arm/readout.json" ]; then
  bash "$L/score.sh" "$sha" "early-$arm" "x=m6/$arm-e13" "k=m6/$ctl-e13" -- "k:x" > /dev/null || exit 1
fi
mkdir -p "$M6/rules"
(cd "$OWN" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$OWN:$OWN/v2/9b" dry python3 -m lux9b.m6_rules early \
  --readout "$M6/early-$arm/readout.json" --arm x --control k --protect "$protect" \
  --output "$M6/rules/early-$arm.json") > /dev/null || exit 1
[ "${DRY_RUN:-0}" = 1 ] && exit 0
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("early", sys.argv[2], "continue=%s" % d["continue"], "P gain %+.2f" % d["P_gain"], d["reasons"])' \
  "$M6/rules/early-$arm.json" "$arm"
