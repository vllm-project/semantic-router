#!/usr/bin/env bash
# ~27B M5 mlx-diag scoring and item-4 pairing on node A, where the mlx-diag gold lives (node A host; CPU only).
# Input: node B's mlx-diag collection of NAME pushed over the private link to /data/dev2/xfer/27b-m5/mlx/NAME
# (m5-tail.sh mlx-push). Steps: copy it to /data/dev2/runs/27b/m5/mlx-diag/NAME and check the predictions' SHA-256
# against node B's list; v2.eval.multilingual_panel score (mlx-diag-v1); v2.06b.m8_scorebias mlx-paired NAME vs
# M4-A20r's scored node A collection (/data/dev2/runs/27b/m4-mlx/M4-A20r-soup): R4 = Choice + Noul type macro, paired
# upper bound >= 0 (XNLI Score reported only). The pairing output is also left in the relay directory for node B.
# Usage: m5-mlx-nodeA.sh MIRROR_SHA NAME
set -euo pipefail
echo "m5 mlx nodeA $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} NAME=${2:?NAME}
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -d "$S/v2/27b/m5" ] || { echo "missing mirror $SHA" >&2; exit 2; }
IN=/data/dev2/xfer/27b-m5/mlx/$NAME OUT=/data/dev2/runs/27b/m5/mlx-diag/$NAME
PANEL=/data/dev2/private/panels/mlx-diag-v1 REF=/data/dev2/runs/27b/m4-mlx/M4-A20r-soup
[ -f "$IN/SHA256SUMS" ] || { echo "no relayed collection $IN" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 66; }
mkdir -p "$(dirname "$OUT")"
cp -a "$IN" "$OUT"
(cd "$OUT" && sha256sum -c --quiet SHA256SUMS)
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1
python3 -m v2.eval.multilingual_panel score --panel "$PANEL" --predictions "$OUT/output/mlx-diag.predictions.jsonl" \
  --output "$OUT/mlx-diag.score.json" > "$OUT/score.log"
python3 -m v2.06b.m8_scorebias mlx-paired --candidate-run "$OUT" --released-run "$REF" --panel "$PANEL" \
  --output "$OUT/../mlx-paired-$NAME-vs-A20r.json"
cp -p "$OUT/../mlx-paired-$NAME-vs-A20r.json" "$IN/../$NAME-vs-A20r.json"
echo "m5 mlx nodeA $NAME complete: $(date -u +%FT%TZ)"
