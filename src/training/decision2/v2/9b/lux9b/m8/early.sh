#!/usr/bin/env bash
# usage: early.sh SHA ARM
# CPU, node A: the preregistered early stop of the distillation arm ARM (D1 or D2) after member 1
# (ARM-m1 vs M7's matched control C-m1, both at alpha 1, read at T = 1). Waits (up to 4 h) for
# ARM-m1-e1's readouts (typed DEV, CSS pilot, HT-DEV v2, PN1 dev) and C-m1-e1's M8 re-read (typed DEV,
# CSS pilot, HT-DEV v2), scores the typed DEV / CSS pilot readouts of both (score.sh early-ARM),
# computes the HT-DEV v2 and PN1 screens (C-m1's PN1 screen is M7's screens/C-m1-e1/pn1.json, read
# against the same reference), reads both runs' final SELECT700 metrics and applies
# lux9b.m8_rules early into /data/dev2/runs/9b/m8/rules/early-ARM.json. Prints the decision; exits 0
# whether ARM continues or stops (early_continue ARM reads the file), non-zero only on a failed step.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; arm=$2
[[ "$arm" =~ ^D[12]$ ]] || { echo "ARM must be D1 or D2" >&2; exit 2; }
S=$(code_dir "$sha")
L=$S/v2/9b/lux9b/m8
for run in "$arm-m1-e1-dev" "$arm-m1-e1-css-pilot" "$arm-m1-e1-ht-dev2" "$arm-m1-e1-pn1-dev" \
  C-m1-e1-dev C-m1-e1-css-pilot C-m1-e1-ht-dev2; do
  wait_ok "$run" 240 || exit 1
done
bash "$L/screens.sh" "$sha" "$arm-m1-e1" ref-ka13 > /dev/null || exit 1
bash "$L/screens.sh" "$sha" C-m1-e1 ref-ka13 > /dev/null || exit 1
[ -f "$M8/early-$arm/readout.json" ] || bash "$L/score.sh" "$sha" "early-$arm" "arm=m8/$arm-m1-e1" "control=m8/C-m1-e1" \
  -- control:arm > /dev/null || exit 1
final() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$1/run/COMPLETE.json"; }
sel() { printf '%s/run/select-step-%07d-metrics.json\n' "$1" "$(final "$1")"; }
mkdir -p "$M8/rules"
[ -f "$M8/rules/early-$arm.json" ] || (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S:$S/v2/9b" dry python3 -m lux9b.m8_rules early \
  --readout "$M8/early-$arm/readout.json" --arm-key arm --control-key control \
  --arm-htdev2 "$M8/screens/$arm-m1-e1/htdev2.json" --control-htdev2 "$M8/screens/C-m1-e1/htdev2.json" \
  --arm-pn1 "$M8/screens/$arm-m1-e1/pn1.json" --control-pn1 "$M7/screens/C-m1-e1/pn1.json" \
  --arm-select "$(sel "$M8/$arm-m1")" --control-select "$(sel "$M7/C-m1")" \
  --output "$M8/rules/early-$arm.json") > "$M8/rules/early-$arm.console" || exit 1
[ "${DRY_RUN:-0}" = 1 ] && exit 0
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("early %s continue=%s" % (sys.argv[2], d["continue"]), d["reasons"])' \
  "$M8/rules/early-$arm.json" "$arm"
