#!/usr/bin/env bash
# usage: screens.sh SHA NAME [REF]
# CPU, node A, host python3 from the mirror SHA: the development screens of the Milestone 8 readout
# NAME (readout.sh with --panel ht-dev2 --panel pn1-dev --mlxdev) against M7's reference readout
# REF (default ref-ka13, M7's re-read of the incumbent K-a13; read-only), into
# /data/dev2/runs/9b/m8/screens/NAME/ (the same screens as M7's):
#   htdev2.json  v2.eval.dev_readout on NAME's ht-dev2 predictions vs the eval track's 9B reference
#                (9b-m4-K-a13; FLAG at a delta <= -0.02)
#   pn1.json     lux9b.m7_rules pn1 vs REF (hop and clean gold-no yes-rates, paired group bootstrap)
#   mlxdev.json  v2.dec.mlx_dev compare REF -> NAME on MLX-DEV-9B (+ mlxdev.score.json)
# A missing prediction file skips its screen (reported); other failures stop.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; name=$2; ref=${3:-ref-ka13}
S=$(code_dir "$sha")
out=$M8/screens/$name
mkdir -p "$out"
py() { (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S:$S/v2/9b" dry python3 -m "$@"); }
ht=$M8/$name-ht-dev2/ht-dev2.predictions.jsonl
pn=$M8/$name-pn1-dev/pn1-dev.predictions.jsonl
ml=$M8/$name-mlxdev/mlxdev-predictions.jsonl
if [ -s "$ht" ]; then
  [ -f "$out/htdev2.json" ] || py v2.eval.dev_readout --htdev2 "$ht" --htdev2-reference "$HTDEV2_REF" \
    --label "$name" --output "$out/htdev2.json" > "$out/htdev2.console"
else echo "screens $name: no ht-dev2 predictions"; fi
if [ -s "$pn" ]; then
  [ -f "$out/pn1.json" ] || py lux9b.m7_rules pn1 --gold "$PN1_GOLD" --predictions "$pn" \
    --reference-predictions "$M7/$ref-pn1-dev/pn1-dev.predictions.jsonl" --label "$name" \
    --reference-label "$ref" --output "$out/pn1.json" > "$out/pn1.console"
else echo "screens $name: no pn1-dev predictions"; fi
if [ -s "$ml" ]; then
  idx=$MLXDEV/panel.jsonl.index.jsonl
  [ -f "$out/mlxdev.score.json" ] || py v2.dec.mlx_dev score --index "$idx" --predictions "$ml" \
    --output "$out/mlxdev.score.json" > "$out/mlxdev.score.console"
  [ -f "$out/mlxdev.json" ] || py v2.dec.mlx_dev compare --index "$idx" \
    --a "$M7/$ref-mlxdev/mlxdev-predictions.jsonl" --b "$ml" --output "$out/mlxdev.json" > "$out/mlxdev.console"
else echo "screens $name: no mlxdev predictions"; fi
echo "screens $name done: $(ls "$out" | tr '\n' ' ')"
