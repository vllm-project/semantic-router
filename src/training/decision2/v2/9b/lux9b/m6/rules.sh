#!/usr/bin/env bash
# usage: rules.sh SHA OUT_NAME
# CPU, node A: the preregistered Milestone 6 development rules on every finished readout.
# score.sh (runtime mirror 3277dec9d) reads the reference ref-ka13, M4's K seeds (K-s1, K-s2, the
# node-A re-read K-s3-A), every M6 seed (-s4, -s5) and every line point (-a13, -a12, -a23 and the soup
# at alpha 1) that has both development predictions, plus M4's K line points (report only) into
# /data/dev2/runs/9b/m6/OUT_NAME/readout.json. lux9b.m6_rules then writes, into m6/rules/OUT_NAME/:
# seed-{KA,KH,K5}.json, alpha-{KA,KH,K5}.json (the rule), alpha-K2.json (report only) and
# finalists.json (priority KA, KH, K5). A line whose arm has no soup readout is skipped.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; out_name=$2
OWN=$(code_dir "$sha")
L=$OWN/v2/9b/lux9b/m6
R=$M6/rules/$out_name
[ ! -e "$R" ] || { echo "$R exists" >&2; exit 66; }
have() { [ -s "$RUNS/$1-dev/dev.predictions.jsonl" ] && [ -s "$RUNS/$1-css-pilot/css-pilot.predictions.jsonl" ]; }
arms=()
add() { if have "$2"; then arms+=("$1=$2"); fi; }
add ref m6/ref-ka13
add k1 m4/K-s1; add k2 m4/K-s2; add k3 m4/K-s3-A
add m4ka13 m4/K-a13; add m4ka12 m4/K-a12; add m4ka23 m4/K-a23; add m4ksoup m4/K-soup
for arm in K KA KH; do
  for s in 4 5; do add "$arm-s$s" "m6/$arm-s$s"; done
done
for line in KA KH K5 K2; do
  for p in a13 a12 a23; do add "$line-$p" "m6/$line-$p"; done
  add "$line-soup" "m6/$line-soup"
done
have m6/ref-ka13 || { echo "no reference readout" >&2; exit 2; }
bash "$L/score.sh" "$sha" "$out_name" "${arms[@]}" -- > /dev/null
mkdir -p "$R"
rules() { (cd "$OWN" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$OWN:$OWN/v2/9b" dry python3 -m lux9b.m6_rules "$@") > /dev/null; }
RO=$M6/$out_name/readout.json
lines=()
for line in KA KH K5; do
  have "m6/$line-soup" || continue
  case $line in
    K5) seeds=k1,k2,k3,K-s4,K-s5; primary=K-s4 ;;
    *) seeds=$line-s4,$line-s5; primary=$line-s4 ;;
  esac
  rules seed --readout "$RO" --soup "$line-soup" --seeds "$seeds" --primary "$primary" --output "$R/seed-$line.json"
  pts=()
  for p in 1/3=a13 1/2=a12 2/3=a23 1=soup; do
    have "m6/$line-${p#*=}" && pts+=(--point "${p%=*}=$line-${p#*=}")
  done
  rules alpha --readout "$RO" --name "$line" --ref ref "${pts[@]}" --output "$R/alpha-$line.json"
  lines+=(--line "$line=$R/alpha-$line.json")
done
if have m6/K2-soup; then
  pts=()
  for p in 1/3=a13 1/2=a12 1=soup; do have "m6/K2-${p#*=}" && pts+=(--point "${p%=*}=K2-${p#*=}"); done
  rules alpha --readout "$RO" --name K2 --ref ref "${pts[@]}" --output "$R/alpha-K2.json"
fi
if [ ${#lines[@]} -gt 0 ]; then
  rules finalists "${lines[@]}" --output "$R/finalists.json"
fi
sha256sum "$R"/*.json "$RO"
