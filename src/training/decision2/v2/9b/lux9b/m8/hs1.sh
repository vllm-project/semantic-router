#!/usr/bin/env bash
# usage: hs1.sh SHA NAME LEFT_RUN RIGHT_RUN
# CPU, node A, host python3: the hs1-dev diagnostic (report only; never a selection criterion):
# v2.data.hs1.validity from the mirror SHA on m8/LEFT_RUN/hs1-dev.predictions.jsonl (left) vs
# m8/RIGHT_RUN/hs1-dev.predictions.jsonl (right; the incumbent), gold on node A, into
# /data/dev2/runs/9b/m8/hs1/NAME.json (per-family accuracy, F1 quote adoption, F3 false yes).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; name=$2; left=$3; right=$4
S=$(code_dir "$sha")
for run in "$left" "$right"; do
  [ -s "$M8/$run/hs1-dev.predictions.jsonl" ] || { echo "no hs1-dev predictions in $run" >&2; exit 2; }
done
mkdir -p "$M8/hs1"
(cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S" dry python3 -m v2.data.hs1.validity \
  --gold "$DATA/dev2/private/panels/gold/hs1-dev.gold.jsonl" \
  --left "$M8/$left/hs1-dev.predictions.jsonl" --left-name "$left" \
  --right "$M8/$right/hs1-dev.predictions.jsonl" --right-name "$right" --output "$M8/hs1/$name.json")
