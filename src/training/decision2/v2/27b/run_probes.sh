#!/usr/bin/env bash
# Sequential backbone probes (native behaviour, cost, frozen features) on one GPU.
# Usage: run_probes.sh GPU MIRROR_SHA KEY [KEY ...]
# Keys: qwen38-27b qwen35-27b gemma4-26b-a4b-it gemma4-26b-a4b
set -euo pipefail

GPU=$1 SHA=$2
shift 2
CODE=/data/dev2/src/$SHA/src/training/decision2
ROOT=/data/dev2/runs/27b/A3-probes
DATA=/data/decision20-20260926/data
BENCH=/data/decision20-20260926/runs
CACHE=/data/dev2/hf-cache
cd "$CODE"

for KEY in "$@"; do
  case $KEY in
    qwen38-27b) MOUNT=/data/decision20-20260926/models/Qwen3.8-27B SUB=. REPO=Qwen/Qwen3.8-27B REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 STAGE=posttrained ;;
    qwen35-27b) MOUNT=$CACHE SUB=models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654 REPO=Qwen/Qwen3.5-27B REV=fc05daec18b0a78c049392ed2e771dde82bdf654 STAGE=posttrained ;;
    gemma4-26b-a4b-it) MOUNT=$CACHE SUB=models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52 REPO=google/gemma-4-26B-A4B-it REV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52 STAGE=posttrained ;;
    gemma4-26b-a4b) MOUNT=$CACHE SUB=models--google--gemma-4-26B-A4B/snapshots/24548b62aa021d562695c04aaf7758a1ea47990b REPO=google/gemma-4-26B-A4B REV=24548b62aa021d562695c04aaf7758a1ea47990b STAGE=base ;;
    *) echo "unknown key $KEY" >&2; exit 2 ;;
  esac
  OUT=$ROOT/$KEY
  mkdir -p "$OUT"
  python3 -m v2.27b.launch --name "d2-27b-probe-$KEY" --gpu "$GPU" --cap-hours 1.5 \
    --purpose "A3/V probe $KEY" --receipt "$OUT/launch.json" \
    --mount "$CODE:/code" --mount "$MOUNT:/source" --mount "$OUT:/out:rw" \
    --mount "$DATA/qwen38_27b_full4096_v1/train.jsonl:/inputs/train.jsonl" \
    --mount "$DATA/rights_clean_goemotions_v2/select.jsonl:/inputs/select.jsonl" \
    --mount "$DATA/rights_clean_goemotions_v2/cal.jsonl:/inputs/cal.jsonl" \
    --mount "$BENCH/dev.prompts.jsonl:/inputs/dev.prompts.jsonl" \
    --mount "$BENCH/css-transfer-v1/css-pilot.prompts.jsonl:/inputs/css-pilot.prompts.jsonl" \
    --env PYTHONPATH=/code -- python3 -m v2.27b.backbone_probe --source "/source/$SUB" \
    --repo-id "$REPO" --revision "$REV" --source-stage "$STAGE" \
    --train /inputs/train.jsonl --select /inputs/select.jsonl --cal /inputs/cal.jsonl \
    --dev /inputs/dev.prompts.jsonl --css-pilot /inputs/css-pilot.prompts.jsonl \
    --output /out/features --receipt /out/probe.json
done
echo "probes complete: $*"
