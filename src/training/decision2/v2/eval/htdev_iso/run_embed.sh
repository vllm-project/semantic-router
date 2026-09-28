#!/usr/bin/env bash
# HT-DEV isolation embedding scan on node A (prereg 3.5(ii), amendment 1 item 5).
#
# Usage: run_embed.sh <mirror-root> prep|scan <gpu> <key>...
#
# prep: CPU, builds <iso>/work/embed/ (training-social protected inventory + one
# candidates file per key). scan: one shared-lease GPU job (<= 30 min, killed after),
# Qwen3-Embedding-0.6B @97b0c614 from /data/dev2/hf-cache, thresholds 0.93 / 0.85.
# The caller writes and closes the lease owner file.
set -euo pipefail
umask 077
SRC="$1"; step="$2"; GPU="$3"; shift 3
S=$SRC/src/training/decision2
P=$S/v2/eval/htdev_iso
ISO=/data/dev2/private/htdev/iso
E=$ISO/work/embed
MODEL=/data/dev2/hf-cache/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
IMG=decision20-train-fast:host2
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ISO/OPERATIONS.log"; }
MOUNTS=(-v "$SRC:$SRC:ro" -v /data/dev2/private/htdev:/data/dev2/private/htdev
  -v /data/dev2/private/eval:/data/dev2/private/eval:ro -v /data/dev2/private/data:/data/dev2/private/data:ro
  -v /data/dev2/runs:/data/dev2/runs:ro -v /data/dev2/hf-cache:/data/dev2/hf-cache:ro
  -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -w "$S")
case "$step" in
  prep)
    args=(); for k in "$@"; do args+=(--source "$k"); done
    log "embed prep start keys=$*"
    docker run --rm --network none "${MOUNTS[@]}" "$IMG" python3 -m v2.eval.htdev_iso.embed_prep \
      --corpora "$ISO/training-corpora.json" --protected-rows "$ISO/work/protected-all-splits.jsonl" \
      "${args[@]}" --families "$P/embed-families.json" --out "$E" --workers 48
    log "embed prep done"
    ;;
  scan)
    args=(); for k in "$@"; do args+=(--candidates "$E/$k.jsonl"); done
    log "embed scan start gpu=$GPU keys=$*"
    start=$(date +%s)
    timeout --kill-after=30 1800 docker run --rm --name "htdev-iso-embed-$$" --network none \
      --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
      -e ROCR_VISIBLE_DEVICES="$GPU" -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
      "${MOUNTS[@]}" "$IMG" python3 -m v2.data.embed_scan "${args[@]}" \
      --protected-inventory "$E/pi/manifest.json" --model-path "$MODEL" --device cuda:0 --batch 256 \
      --quarantine 0.93 --review 0.85 --sample 20 --seed decision2-embed-scan-v1 \
      --private-receipt "$E/embed.private.json" --public-receipt "$E/embed.public.json" > "$E/embed.stdout" 2>&1
    log "embed scan done gpu=$GPU wall_seconds=$(( $(date +%s) - start ))"
    ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
