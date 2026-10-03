#!/usr/bin/env bash
# Decoder M17 data preparation on node E / F (prereg dec-m17-prereg-2026-10-02.md, "Data" and "Development readouts"),
# CPU only (host python3, standard library), from an exact mirror. Idempotent; a failed build is not rerun.
#   1. inputs, each against its hash: LH's released TRAIN (M10), M12 4b-LHA10 / 4b-LHA TRAIN and ids files, M13's 4B
#      SD targets (M15's copy, m15/inputs/teacher-4b.jsonl);
#   2. TRAIN + teacher of both arms (m17_data.py, sentfin dropped, seed 20261002), on both nodes (the data lock
#      compares the builds);
#   3. node F: Triton caches 4b-train / 4b-read (cp -a copies of this node's M15 4B caches), the reference readouts
#      4b-LH-f and 4b-LHA10SD-m13 (copied from M15, all panels and the old MLX-DEV read) and the old MLX-DEV 4B panel
#      (M15's, report only).
#
# usage: M17_NODE=e|f m17-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M17_NODE:?set M17_NODE=e or f}
R=/data/dev2/runs/dec
M=$R/m17
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m17
mkdir -p "$M/data" "$M/lines" "$M/logs"
log() { echo "$(date -u +%FT%TZ) prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
BASE=$R/m10/data/m10-4b-base/train.jsonl BASE_SHA=c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09
A10=$R/m12/data/4b/4b-LHA10 A10_SHA=d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5
A10_IDS_SHA=e429ce9b7506e3e39340e6c2f8dd7804ea65cb7469367e71f8257bc5b4ac064e
A25=$R/m12/data/4b/4b-LHA A25_SHA=e9d30c8f89ce215ff4e1b5f57e5eafd194c1822d1108c63ba5c52e7d119fd810
A25_IDS_SHA=c7193bd82c56ba6dcd303ed3b1cc6c7b0e0a9710857a0ffccc861bbb3c729c8a
TEACHER=$R/m15/inputs/teacher-4b.jsonl TEACHER_SHA=7639fab17c719bb3ed7a18bd16397130f109c8bd204d17c45f6743d6f0c17496
SEED=20261002

check "$BASE" $BASE_SHA
check "$A10/train.jsonl" $A10_SHA
check "$A10/train.ids.jsonl" $A10_IDS_SHA
check "$A25/train.jsonl" $A25_SHA
check "$A25/train.ids.jsonl" $A25_IDS_SHA
check "$TEACHER" $TEACHER_SHA
log "M17 inputs verified"

if [ -f "$M/data/4b/report.json" ]; then
  log "TRAIN already built"
elif [ -e "$M/data/4b.FAILED" ]; then
  log "build failed earlier; not rerun"
  exit 1
elif python3 -B "$OPS/m17_data.py" --base "$BASE" --base-sha $BASE_SHA --base-train "$A10/train.jsonl" \
  --base-train-sha $A10_SHA --base-ids "$A10/train.ids.jsonl" --pool-train "$A25/train.jsonl" --pool-sha $A25_SHA \
  --pool-ids "$A25/train.ids.jsonl" --teacher "$TEACHER" --teacher-sha $TEACHER_SHA --drop sentfin \
  --arm 4b-LHS10SD=0.10 --arm 4b-LHS17SD=0.17 --seed $SEED --output "$M/data/4b" > "$M/data/4b.log" 2>&1; then
  log "TRAIN built: $(tail -1 "$M/data/4b.log" | cut -c1-900)"
else
  touch "$M/data/4b.FAILED"
  log "FAILED: TRAIN build (see $M/data/4b.log)"
  exit 1
fi

[ "$NODE" = f ] || { log "prep finished (node E: data build only)"; exit 0; }
mkdir -p "$M/triton-cache"
for c in 4b-train 4b-read; do
  [ -d "$M/triton-cache/$c" ] && continue
  [ -d "$R/m15/triton-cache/$c" ] || { log "FAILED: no M15 cache $c on this node"; exit 1; }
  cp -a "$R/m15/triton-cache/$c" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
  log "Triton cache $c copied from m15/triton-cache/$c ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
done
for ref in 4b-LH-f 4b-LHA10SD-m13; do
  [ -d "$M/lines/$ref" ] && continue
  cp -a "$R/m15/lines/$ref" "$M/lines/$ref.tmp" && mv "$M/lines/$ref.tmp" "$M/lines/$ref"
  log "readouts $ref copied from m15/lines/$ref ($(find "$M/lines/$ref" -name '*.predictions.jsonl' | wc -l) panels)"
done
if [ ! -d "$M/mlxdev/4b" ]; then
  mkdir -p "$M/mlxdev"
  cp -a "$R/m15/mlxdev/4b" "$M/mlxdev/4b.tmp" && mv "$M/mlxdev/4b.tmp" "$M/mlxdev/4b"
  log "old MLX-DEV 4B panel copied from m15/mlxdev/4b ($(sha "$M/mlxdev/4b/panel.jsonl" | cut -c1-16))"
fi
log "prep finished"
