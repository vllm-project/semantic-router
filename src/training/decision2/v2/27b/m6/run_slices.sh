#!/usr/bin/env bash
# ~27B M6 kernel-path development slices of one checkpoint (node B host side; one GPU of the M6 allocation, launch3
# m6-b, track 27b; never a release score): raw T = 1 probabilities on SELECT-format row files (PN1 dev, IB DEV) through
# `kernel_readout slices`, 32,768 tokens, on a fresh verified copy of DEV2.0-27B's scored cache 03b172f1.
# Usage: run_slices.sh NAME CKPT GPU MIRROR_SHA SLICE...    SLICE = NAME=HOST_ROWS=SHA256
#   -> /data/dev2/runs/27b/m6/slices/NAME/{probs/<slice>.probs.jsonl, probs/slices.json, receipts/slices.json}
# CHECKPOINT_FORMAT: peft-lora/1 (default; LoRA soups) or full. DRY_RUN=1: see kernel_common.sh.
set -euo pipefail
echo "m6 slices $*: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU} MIRROR_SHA=${4:?MIRROR_SHA}
shift 4
[ $# -ge 1 ] || { echo "at least one SLICE (NAME=HOST_ROWS=SHA256)" >&2; exit 2; }
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
[[ "$MIRROR_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
SRC=$MIRROR_SHA
[ -d "/data/dev2/src/$SRC" ] || SRC=$MIRROR_SHA-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
LIMIT=32768
FROZEN=/data/dev2/runs/27b/m3-f2/f1-scored-cache
CACHE_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
CHECKPOINT_FORMAT=${CHECKPOINT_FORMAT:-peft-lora/1}
case "$CHECKPOINT_FORMAT" in full | peft-lora/1) ;; *) echo "CHECKPOINT_FORMAT is full or peft-lora/1" >&2; exit 2 ;; esac
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp DEV2_27B_LAUNCH_ALLOC=m6-b
source "$S/v2/27b/kernel_common.sh"
source "$S/v2/27b/m4b/common3.sh"
OUT=/data/dev2/runs/27b/m6/slices/$NAME
mounts=() args=() paths=()
for spec in "$@"; do
  slice=${spec%%=*} rest=${spec#*=}
  path=${rest%=*} sha=${rest##*=}
  [[ "$slice" =~ ^[A-Za-z0-9._-]+$ ]] && [[ "$sha" =~ ^[0-9a-f]{64}$ ]] && [ -n "$path" ] ||
    { echo "bad SLICE $spec (NAME=HOST_ROWS=SHA256)" >&2; exit 2; }
  [ "$(sha256sum < "$path" | cut -c1-64)" = "$sha" ] || { echo "$path is not $sha" >&2; exit 2; }
  paths+=("$path")
  mounts+=(--mount "$path:/data/slice-$slice.jsonl")
  args+=(--slice "$slice=/data/slice-$slice.jsonl=$sha")
done
verify_mirror "$MIRROR_SHA"
need "$CKPT" "$BASE" "$FROZEN" "${paths[@]}"
python3 -m v2.27b.m4b.ckpt_format check --checkpoint "$CKPT" --format "$CHECKPOINT_FORMAT"
verify_cache "$FROZEN" "$CACHE_SHA"
[ "$DRY_RUN" = 1 ] || verify_lease
[ ! -e "$OUT/probs" ] || { echo "$OUT/probs exists: one slices run per NAME" >&2; exit 66; }
mkdir -p "$OUT/receipts"
cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/triton-cache"
status=0
launcher "d2-27b-m6-$NAME-slices" 2.0 "M6 $NAME kernel slices at $LIMIT" "$OUT/receipts/slices.json" \
  "$OUT/triton-cache" -- --mount "$CKPT:$CKPT" "${mounts[@]}" --mount "$OUT:$OUT:rw" -- \
  python3 -m v2.27b.kernel_readout slices --checkpoint "$CKPT" --source-path "$BASE" --max-length "$LIMIT" \
  "${args[@]}" --out-dir "$OUT/probs" || status=$?
cache_finish "$OUT/triton-cache"
[ "$status" = 0 ] || exit "$status"
echo "m6 slices $NAME complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
