#!/usr/bin/env bash
# HT-DEV isolation, CPU part on node A: prepare sources, freeze manifest, names, rows, overlap.
#
# Usage: run_lexical.sh <mirror-root> <step>...   steps: prepare manifest rows names overlap
#
# <mirror-root> is /data/dev2/src/<sha>-src_training_decision2 of a pushed commit. Every
# Python step runs in decision20-train-fast:host2 with --network none; outputs are written
# O_EXCL under /data/dev2/private/htdev/iso (mode 700). Counts only on stdout.
set -euo pipefail
umask 077
SRC="$1"; shift
S=$SRC/src/training/decision2
P=$S/v2/eval/htdev_iso
ISO=/data/dev2/private/htdev/iso
W=$ISO/work
R2=/data/dev2/runs/eval/m4/c1-event2-recheck2
C1T=/data/dev2/private/eval/c1-corpora/training
mkdir -p "$W"
RUN=(docker run --rm --network none -v "$SRC:$SRC:ro"
  -v /data/dev2/private/htdev:/data/dev2/private/htdev
  -v /data/dev2/private/eval:/data/dev2/private/eval:ro
  -v /data/dev2/private/data:/data/dev2/private/data:ro
  -v /data/dev2/runs:/data/dev2/runs:ro -v /data/dev2/src:/data/dev2/src:ro
  -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 -w "$S" decision20-train-fast:host2)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ISO/OPERATIONS.log"; }
keys=$(python3 -c 'import json,sys; print(" ".join(k for k,v in json.load(open(sys.argv[1]))["sources"].items() if v["origin"]!="unavailable"))' "$P/sources.json")
TRAIN_LABELS=(
  "hf-head=$ISO/training/hf/head-75e557f1" "hf-hist=$ISO/training/hf/hist"
  "rev-5c0255ed=$C1T/5c0255ed" "rev-39a120ca=$C1T/39a120ca" "rev-ed87a03a=$C1T/ed87a03a"
  "rev-d8eae3e4=$C1T/d8eae3e4" "rev-3a99bf1c=$C1T/3a99bf1c" "rev-12912429=$C1T/12912429"
  "r2-hf-delta=$R2/hf-delta" "r2-extra-derived=$R2/extra-derived" "r2-nodeB=$R2/nodeB-snapshot"
  "r2-local=$R2/local-snapshot" "r2-local-code=$R2/local-snapshot-code"
  "local-code=$ISO/training/local-code"
)
for d in "$ISO"/training/local/*/; do d=${d%/}; TRAIN_LABELS+=("local-${d##*/}=$d"); done
for step in "$@"; do
  log "step $step start (code $(basename "$SRC"))"
  case "$step" in
    prepare)
      "${RUN[@]}" python3 -m v2.eval.htdev_iso.prepare --spec "$P/sources.json" --iso "$ISO"
      ;;
    manifest)
      args=(); for l in "${TRAIN_LABELS[@]}"; do args+=(--label "$l"); done
      "${RUN[@]}" python3 -m v2.eval.htdev_iso.manifest "${args[@]}" \
        --merge-old /data/dev2/private/eval/c1-corpora/manifest.json --merge-glob 'train-*' \
        --output "$ISO/TRAINING-MANIFEST.json" --corpora "$ISO/training-corpora.json" --workers 48
      ;;
    rows)
      args=(); for k in $keys; do args+=(--source "$k"); done
      "${RUN[@]}" python3 -m v2.eval.sealed.independence rows --sources-dir "$ISO/sources" \
        "${args[@]}" --output "$W/protected-all-splits.jsonl"
      ;;
    names)
      args=()
      for l in "${TRAIN_LABELS[@]}"; do args+=(--root "${l#*=}"); done
      for m in /data/dev2/src/{e0ac90ef8,2d0e69965,f344f1aec,f0c78cc99,9d90212dd,879ec5e4e,313dafe98,9b70a44a8,73398871f,cae64f4e8,be472b957}*-src_training_decision2; do
        args+=(--root "$m")
      done
      args+=(--root /data/dev2/runs/a7 --root /data/dev2/runs/release/scratch/d1cards
        --root /data/dev2/private/eval/c1-corpora/aggregators)
      "${RUN[@]}" python3 -m v2.eval.sealed.independence names --terms "$P/htdev-source-terms.json" \
        "${args[@]}" --output "$W/names.json" | tail -c 1500
      ;;
    overlap)
      "${RUN[@]}" python3 -m v2.eval.sealed.overlap scan --protected "$W/protected-all-splits.jsonl" \
        --manifest "$ISO/training-corpora.json" --workers 48 --exact-min-tokens 8 \
        --output "$W/overlap-receipt.json" --hits "$W/overlap-hits.jsonl"
      ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
  log "step $step done"
done
