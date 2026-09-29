#!/usr/bin/env bash
# DEV2.0-26B private upload + verification (no --collect) on node B GPU6, shared lease owner.release.
# Usage (node B): bash <mirror>/v2/release/records/dev2-26b-release-2026-09-29/ops/release-upload.sh <SRC>
#   SRC = <commit>-src_training_decision2 under /data/dev2/src (the commit that holds the spec).
set -euo pipefail
SRC=$1
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=/data/dev2/src/$SRC/src/training/decision2
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
G=/data/dev2/private/panels/goldfree
P=/data/dev2/runs/27b/M3-A-soup/formal/output
HFC=/data/dev2/hf-cache
BASE_REPO=$HFC/models--Qwen--Qwen3.8-27B
BASE=$BASE_REPO/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
FROZEN=/data/dev2/runs/27b/M3-A-soup/formal/triton-cache
FROZEN_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
TC=/data/dev2/runs/release/triton/dev2-26b-f1-copy-$TS
W=/data/dev2/runs/release/dev2-26b-release-$TS
HFPY=/data/dev2/tools/hf-cli/bin/python
REPO=llm-semantic-router/DEV2.0-26B
GPU=6
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton

# GPU6 belongs to the ~27B track: run only while it is idle and not marked running.
use=$(rocm-smi -d "$GPU" --showuse | awk -F': ' '/GPU use/ {print $NF+0}')
vram=$(rocm-smi -d "$GPU" --showmemuse | awk -F': ' '/VRAM%/ {print $NF+0}')
if [[ "$use" != 0 || "$vram" != 0 ]] || grep -q '"status": "running"' /data/dev2/leases/gpu$GPU.lock/owner; then
  echo "gpu$GPU is busy (use $use%, vram $vram%) or its owner is running" >&2
  exit 1
fi
trap 'rm -f /data/dev2/leases/gpu$GPU.lock/owner.release' EXIT

bash "$S/v2/common/hf_headroom.sh" --min-free-gb 1
cd "$S"
python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$FROZEN_SHA" --dest "$TC"
set -x
status=0
"$S/v2/release/release.sh" --spec "$S/v2/release/specs/dev2-26b-release.json" --src "$SRC" --work "$W" \
  --image "$IMAGE" --gpu "$GPU" --track release --shared-lease release --threads 4 \
  --site /opt/decision-fla --require-kernels --base-path "$BASE" \
  --env HIP_FORCE_DEV_KERNARG=1 --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" \
  --env "HF_HUB_CACHE=$HFC" --mount "$BASE_REPO" \
  --mount "$G" --mount "$P" \
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600" \
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547" \
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231" \
  --upload || status=$?
set +x
python3 -m v2.27b.triton_cache finish --dest "$TC" || true
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }

REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
mkdir -p "$W/extra"
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-26B" --output "$W/extra/card-http.json"
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-26B" --output "$W/extra/hub-links.json"
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || true
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV cache=$TC"
