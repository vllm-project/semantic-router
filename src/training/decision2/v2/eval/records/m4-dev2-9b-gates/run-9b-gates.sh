#!/usr/bin/env bash
# usage (node A, CPU only): bash run-9b-gates.sh MIRROR_SHA
# 9B release candidate K-a13 from stored sealed post-key predictions: paired v3 / H / T intervals
# and the type-collapse check (v2.eval.gates), then the rescreen-overlap exposure of the K soup's
# training file and the with/without-flagged-items rescore (v2.eval.overlap_effects). No GPU, no C1.
set -euo pipefail
sha=$1
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
[ -d "$S" ] || { echo "mirror $S missing" >&2; exit 2; }
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1
R=/data/dev2/runs
G=$R/eval/m4/gates-9b
O=$R/eval/m5/overlap-effects
F=$O/final-9b
CAND=$R/9b/formal-m4/K-a13-16k
X60=$R/9b/m4/data/m4-k-xl-r2-60m/build/train.jsonl
X60_SHA=a66131b1165513128e5a51af78587c4ddefcc836773e724c4514a592e33fc0e0
[ ! -e "$G" ] && [ ! -e "$F" ] || { echo "$G or $F exists" >&2; exit 66; }
mkdir -p "$G" "$F"
cd "$S"

date -u +%FT%TZ > "$G/run.start"
while read -r key name run; do
  python3 -m v2.eval.gates paired --left "$CAND" --right "$run" --left-name K-a13 \
    --right-name "$name" --output "$G/paired-vs-$key.json" > "$G/paired-vs-$key.log"
done <<EOF
lux1 Lux1 $R/eval/m1/d1-lux1-autotune-cache
lux1-shared Lux1-same-renderer $R/9b/formal-m3/lux1-16k-shared
nimble2 Nimble-v2 $R/eval/m2/q6-nimble2
jpt9b JPT-9B $R/eval/m1-adopt/jpt9b
EOF
while read -r key name run; do
  python3 -m v2.eval.gates types --run "$run" --label "$name" \
    --output "$G/types-$key.json" > "$G/types-$key.log"
done <<EOF
cand K-a13 $CAND
lux1 Lux1 $R/eval/m1/d1-lux1-autotune-cache
lux1-shared Lux1-same-renderer $R/9b/formal-m3/lux1-16k-shared
nimble2 Nimble-v2 $R/eval/m2/q6-nimble2
jpt9b JPT-9B $R/eval/m1-adopt/jpt9b
EOF
date -u +%FT%TZ > "$G/run.end"

date -u +%FT%TZ > "$F/run.start"
python3 -m v2.eval.overlap_effects exposure --groups "$O/final/excluded-groups.json" \
  --train "$X60" --expect-sha256 "$X60_SHA" --label "9B K-a13: K soup training file (m4-k-xl-r2-60m x60)" \
  --output "$F/exposure-9b-k-a13-x60.json" > "$F/exposure.log"
set +e
python3 -m v2.eval.overlap_effects run --spec "$S/v2/eval/records/m4-dev2-9b-gates/overlap-spec-9b.json" \
  --flagged "$O/final/flagged.json" --output "$F/overlap-effects.json" --jobs 24 > "$F/run.log" 2>&1
echo $? > "$F/run.exit"
set -e
date -u +%FT%TZ > "$F/run.end"
exit "$(cat "$F/run.exit")"
