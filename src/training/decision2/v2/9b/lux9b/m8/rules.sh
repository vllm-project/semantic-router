#!/usr/bin/env bash
# usage: rules.sh SHA OUT_NAME
# CPU, node A: the preregistered Milestone 8 development rules on every finished readout.
# score.sh (runtime mirror 3277dec9d) reads the reference ref (M7's re-read of K-a13), M6's re-read
# of K-a13 and K5-a12 (report only), the control line C5 (M7; report only) and every M8 line point
# (KD1 / KD2 / KDX at a13, a12, a23) that has both development predictions into
# /data/dev2/runs/9b/m8/OUT_NAME/readout.json, with the D - C contrasts at equal alpha. M7's rule
# (lux9b.m7_rules alpha: type, family and rule_precedence floors, HT-DEV v2 not FLAG, PN1 dev guard,
# MLX-DEV-9B guard, G* >= 0.01, alpha* = the smallest eligible alpha with G >= 3/4 G*, proxy drop)
# then writes, into m8/rules/OUT_NAME/: alpha-KD1.json, alpha-KD2.json, alpha-KDX.json (each point's
# screens from m8/screens/POINT/), alpha-C5.report.json (the control line under the same rule,
# report only, screens from m7/screens/) and finalists.json (priority KD1, KD2, KDX; at most three).
# A line without a readout is skipped.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; out_name=$2
OWN=$(code_dir "$sha")
L=$OWN/v2/9b/lux9b/m8
R=$M8/rules/$out_name
[ ! -e "$R" ] || { echo "$R exists" >&2; exit 66; }
have() { [ -s "$RUNS/$1-dev/dev.predictions.jsonl" ] && [ -s "$RUNS/$1-css-pilot/css-pilot.predictions.jsonl" ]; }
arms=(); compares=(ref:m6ref ref:k5a12)
add() { if have "$2"; then arms+=("$1=$2"); fi; }
add ref m7/ref-ka13
add m6ref m6/ref-ka13; add k5a12 m6/K5-a12
for p in a13 a12 a23; do add "C5-$p" "m7/C5-$p"; done
for line in KD1 KD2 KDX; do
  for p in a13 a12 a23; do
    add "$line-$p" "m8/$line-$p"
    if have "m8/$line-$p" && have "m7/C5-$p"; then compares+=("C5-$p:$line-$p"); fi
  done
done
have m7/ref-ka13 || { echo "no reference readout" >&2; exit 2; }
bash "$L/score.sh" "$sha" "$out_name" "${arms[@]}" -- "${compares[@]}" > /dev/null
mkdir -p "$R"
rules() { (cd "$OWN" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$OWN:$OWN/v2/9b" dry python3 -m lux9b.m7_rules "$@") > /dev/null; }
RO=$M8/$out_name/readout.json
line_rule() {  # LINE RUNS_SUBDIR (m7 or m8) OUTPUT
  local line=$1 sub=$2 out=$3 pts=() scr=() key d p
  for p in 1/3=a13 1/2=a12 2/3=a23; do
    key=$line-${p#*=}
    have "$sub/$key" || continue
    pts+=(--point "${p%=*}=$key")
    d=$RUNS/$sub/screens/$key
    if [ -f "$d/htdev2.json" ] && [ -f "$d/pn1.json" ] && [ -f "$d/mlxdev.json" ]; then
      scr+=(--screen "$key=$d/htdev2.json,$d/pn1.json,$d/mlxdev.json")
    fi
  done
  [ ${#pts[@]} -gt 0 ] || return 1
  rules alpha --readout "$RO" --name "$line" --ref ref "${pts[@]}" "${scr[@]}" --output "$out"
}
line_rule C5 m7 "$R/alpha-C5.report.json" || true
lines=()
for line in KD1 KD2 KDX; do
  if line_rule "$line" m8 "$R/alpha-$line.json"; then lines+=(--line "$line=$R/alpha-$line.json"); fi
done
if [ ${#lines[@]} -gt 0 ]; then
  rules finalists "${lines[@]}" --output "$R/finalists.json"
fi
sha256sum "$R"/*.json "$RO"
