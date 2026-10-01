#!/usr/bin/env bash
# ~27B M6 on node A (host side; CPU only), where the mlx-diag gold lives: wait until node B has pushed NAME's mlx-diag
# collection (`m6-tail.sh mlx-push`, then the marker /data/dev2/xfer/27b-m6/mlx/NAME.PUSHED), copy it to
# /data/dev2/runs/27b/m6/mlx-diag/NAME, check the predictions against node B's SHA-256 list, score it
# (v2.eval.multilingual_panel score, mlx-diag-v1) and pair it with A20r's scored node A collection
# (v2.06b.m8_scorebias mlx-paired: R4 = card-eligible Choice + Noul, paired upper bound >= 0; XNLI Score report only).
# The pairing JSON is left in the relay directory for node B's `m6-tail.sh mlx-pull`. mlx/NAME.SKIP (node B's chain
# made no formal run for NAME) ends the wait. Usage: m6-mlx-watch.sh MIRROR_SHA NAME. Detached, log on stdout.
set -euo pipefail
echo "m6 mlx watch $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} NAME=${2:?NAME}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m6/m6-mlx-watch.sh" ] || { echo "missing mirror $SHA" >&2; exit 2; }
X=/data/dev2/xfer/27b-m6/mlx
mkdir -p "$X"
while :; do
  if [ -f "$X/$NAME.SKIP" ]; then
    echo "$(date -u +%FT%TZ) no mlx-diag collection will come: $(cat "$X/$NAME.SKIP")"
    exit 0
  fi
  [ -f "$X/$NAME.PUSHED" ] && break
  sleep 300
done
IN=$X/$NAME OUT=/data/dev2/runs/27b/m6/mlx-diag/$NAME
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
cp -p "$OUT/../mlx-paired-$NAME-vs-A20r.json" "$X/$NAME-vs-A20r.json"
echo "m6 mlx watch $NAME complete: $(date -u +%FT%TZ)"
