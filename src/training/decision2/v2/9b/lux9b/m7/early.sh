#!/usr/bin/env bash
# usage: early.sh SHA [ARM]
# CPU, node A: the preregistered early stop of the treatment arm ARM (default P; amendment 1 adds Q)
# after the first member (ARM-m1 vs its matched control C-m1, both at alpha 1). Waits (up to 4 h) for
# both PN1 dev readouts (readout.sh --no-cal --panel pn1-dev: ARM-m1-e1, C-m1-e1) and the reference's
# (ref-ka13), scores each against the reference (screens.sh), reads both runs' final SELECT700
# metrics, and applies lux9b.m7_rules early into /data/dev2/runs/9b/m7/rules/early-ARM.json. Prints
# the decision; exits 0 whether ARM continues or stops (early_continue ARM reads the file), non-zero
# only on a failed step.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; arm=${2:-P}
S=$(code_dir "$sha")
L=$S/v2/9b/lux9b/m7
for run in ref-ka13-pn1-dev "$arm-m1-e1-pn1-dev" C-m1-e1-pn1-dev; do
  wait_ok "$run" 240 || exit 1
done
for run in "$arm-m1-e1" C-m1-e1; do bash "$L/screens.sh" "$sha" "$run" ref-ka13 > /dev/null || exit 1; done
final() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$M7/$1/run/COMPLETE.json"; }
sel() { printf '%s/%s/run/select-step-%07d-metrics.json\n' "$M7" "$1" "$(final "$1")"; }
mkdir -p "$M7/rules"
[ -f "$M7/rules/early-$arm.json" ] || (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S:$S/v2/9b" dry python3 -m lux9b.m7_rules early \
  --p-pn1 "$M7/screens/$arm-m1-e1/pn1.json" --c-pn1 "$M7/screens/C-m1-e1/pn1.json" \
  --p-select "$(sel "$arm-m1")" --c-select "$(sel C-m1)" --output "$M7/rules/early-$arm.json") > "$M7/rules/early-$arm.console" || exit 1
[ "${DRY_RUN:-0}" = 1 ] && exit 0
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("early %s continue=%s" % (sys.argv[2], d["continue"]), d["P"], d["C"], d["reasons"])' \
  "$M7/rules/early-$arm.json" "$arm"
