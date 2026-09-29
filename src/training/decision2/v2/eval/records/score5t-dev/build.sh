#!/usr/bin/env bash
# Score5-typed-DEV v1 build on node A (CPU only) from an exact mirror:
#
#   bash build.sh <sha>-src_training_decision2
#
# Training-corpora scan -> scan-v1/, panel build -> build-v1/, install of the gold-free
# prompts (0644) and gold (0600) into the panel root, option-key/position leak audit
# on the installed files, SHA256SUMS. Never overwrites; the whole output is logged to
# /data/dev2/runs/eval/score5t-dev/build-v1.log.
# Rules: v2/eval/records/score5t-dev-prereg-2026-09-29.md.
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: build.sh <sha>-src_training_decision2" >&2; exit 2; }
MIRROR=/data/dev2/src/$1
S=$MIRROR/src/training/decision2
[[ -f $MIRROR/.dev2-mirror.json && -d $S ]] || { echo "no mirror at $MIRROR" >&2; exit 1; }

ROOT=/data/dev2/runs/eval/score5t-dev
PANELS=/data/dev2/private/panels
CORPORA=/data/dev2/private/htdev/iso/training-corpora.json
SELECT=$PANELS/gold/select.jsonl
CAL=$PANELS/gold/cal.jsonl
CAL698=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
WORKERS=24
SCAN_DIR=$ROOT/scan-v1
SCAN=$SCAN_DIR/training-scan.json
BUILD=$ROOT/build-v1
LOG=$ROOT/build-v1.log
PROMPTS=(score5t-dev.prompts.jsonl score5t-dev.fit.prompts.jsonl score5t-dev.check.prompts.jsonl)
GOLD=(score5t-dev.gold.jsonl score5t-dev.fit.gold.jsonl score5t-dev.check.gold.jsonl)

mkdir -p "$ROOT"
chmod 700 "$ROOT"
for path in "$LOG" "$SCAN_DIR" "$BUILD"; do
  [[ ! -e $path ]] || { echo "$path exists; refusing to overwrite" >&2; exit 1; }
done
for name in "${PROMPTS[@]}"; do
  [[ ! -e $PANELS/goldfree/$name ]] || { echo "$PANELS/goldfree/$name exists" >&2; exit 1; }
done
for name in "${GOLD[@]}"; do
  [[ ! -e $PANELS/gold/$name ]] || { echo "$PANELS/gold/$name exists" >&2; exit 1; }
done
install -m 600 /dev/null "$LOG"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH=$S CUDA_VISIBLE_DEVICES=""
cd "$S"
started=$(date +%s)
step() { echo "== $(date -u +%FT%TZ) $*"; }

step "mirror $1"
cat "$MIRROR/.dev2-mirror.json"
echo
python3 --version

step "scan ($WORKERS workers)"
mkdir -m 700 "$SCAN_DIR"
timeout 3h nice -n 10 python3 -m v2.eval.score5t scan \
  --manifest "$CORPORA" --output "$SCAN" --workers "$WORKERS"

step "build"
timeout 1h python3 -m v2.eval.score5t build \
  --output-dir "$BUILD" \
  --panels-root "$PANELS" \
  --select "$SELECT" \
  --cal "$CAL" \
  --cal698 "$CAL698" \
  --training-scan "$SCAN"

step "install"
INSTALLED=()
for name in "${PROMPTS[@]}"; do
  install -m 0644 -T "$BUILD/$name" "$PANELS/goldfree/$name"
  cmp "$BUILD/$name" "$PANELS/goldfree/$name"
  INSTALLED+=("$PANELS/goldfree/$name")
done
for name in "${GOLD[@]}"; do
  install -m 0600 -T "$BUILD/$name" "$PANELS/gold/$name"
  cmp "$BUILD/$name" "$PANELS/gold/$name"
  INSTALLED+=("$PANELS/gold/$name")
done
stat -c '%a %U:%G %s %n' "${INSTALLED[@]}"

step "leak audit"
timeout 2h python3 -m v2.eval.leak_audit audit \
  --panel-root "$PANELS" \
  --panel score5t-dev \
  --files "score5t-dev=$PANELS/goldfree/${PROMPTS[0]}:$PANELS/gold/${GOLD[0]}" \
  --output "$BUILD/leak-audit.json"

step "sha256"
install -m 600 /dev/null "$BUILD/SHA256SUMS"
(
  cd "$BUILD"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort | xargs sha256sum
  sha256sum "$SCAN" "${INSTALLED[@]}"
) | tee -a "$BUILD/SHA256SUMS"

step "done in $(($(date +%s) - started)) s"
