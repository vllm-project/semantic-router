#!/usr/bin/env bash
# usage: rules.sh SHA OUT_NAME
# CPU, node A: the preregistered Milestone 7 development rules on every finished readout.
# score.sh (runtime mirror 3277dec9d) reads the reference ref-ka13 (the M7 re-read of K-a13), M6's
# re-read of K-a13 and K5-a12 (report only) and every line point (P5 / C5 at a13, a12, a23) that has
# both development predictions into /data/dev2/runs/9b/m7/OUT_NAME/readout.json. lux9b.m7_rules
# then writes, into m7/rules/OUT_NAME/: alpha-P5.json, alpha-Q5.json (amendment 1) and alpha-C5.json
# (the rule, with each point's screens from m7/screens/POINT/) and finalists.json (priority P5, Q5,
# C5). A line without a readout is skipped.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; out_name=$2
OWN=$(code_dir "$sha")
L=$OWN/v2/9b/lux9b/m7
R=$M7/rules/$out_name
[ ! -e "$R" ] || { echo "$R exists" >&2; exit 66; }
have() { [ -s "$RUNS/$1-dev/dev.predictions.jsonl" ] && [ -s "$RUNS/$1-css-pilot/css-pilot.predictions.jsonl" ]; }
arms=()
add() { if have "$2"; then arms+=("$1=$2"); fi; }
add ref m7/ref-ka13
add m6ref m6/ref-ka13; add k5a12 m6/K5-a12
for line in P5 Q5 C5; do
  for p in a13 a12 a23; do add "$line-$p" "m7/$line-$p"; done
done
have m7/ref-ka13 || { echo "no reference readout" >&2; exit 2; }
bash "$L/score.sh" "$sha" "$out_name" "${arms[@]}" -- ref:m6ref ref:k5a12 > /dev/null
mkdir -p "$R"
rules() { (cd "$OWN" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$OWN:$OWN/v2/9b" dry python3 -m lux9b.m7_rules "$@") > /dev/null; }
RO=$M7/$out_name/readout.json
lines=()
for line in P5 Q5 C5; do
  pts=(); scr=()
  for p in 1/3=a13 1/2=a12 2/3=a23; do
    key=$line-${p#*=}
    have "m7/$key" || continue
    pts+=(--point "${p%=*}=$key")
    d=$M7/screens/$key
    if [ -f "$d/htdev2.json" ] && [ -f "$d/pn1.json" ] && [ -f "$d/mlxdev.json" ]; then
      scr+=(--screen "$key=$d/htdev2.json,$d/pn1.json,$d/mlxdev.json")
    fi
  done
  [ ${#pts[@]} -gt 0 ] || continue
  rules alpha --readout "$RO" --name "$line" --ref ref "${pts[@]}" "${scr[@]}" --output "$R/alpha-$line.json"
  lines+=(--line "$line=$R/alpha-$line.json")
done
if [ ${#lines[@]} -gt 0 ]; then
  rules finalists "${lines[@]}" --output "$R/finalists.json"
fi
sha256sum "$R"/*.json "$RO"
