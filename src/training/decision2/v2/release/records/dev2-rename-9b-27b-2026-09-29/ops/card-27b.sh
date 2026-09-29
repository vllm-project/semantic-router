#!/usr/bin/env bash
# DEV2.0-27B card-only revision after the move from DEV2.0-26B (user directive 2026-09-29 16:05 UTC+8): release.sh
# --upload --collect --already-collected from this mirror (spec dev2-27b-release.json, final decision
# DEV2.0-27B.decision.json), subset parity (typed-final 200, css15 300, public231 100) with a verified copy of F1's
# frozen autotune cache, on one node-B GPU under the shared lease owner.rename-27b. GPU0-2 hold ~27B Milestone 4b and
# are never used. Afterwards: every weight file byte-identical to the released 6931828d, collection order, card HTTP,
# links, storage.
# Usage (node B): bash <mirror>/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops/card-27b.sh [--preview] [--gpu N]
#   --preview: CPU build only, for the card review before any upload (no GPU, no Hub writes).
set -euo pipefail
preview=0 gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preview) preview=1; shift ;;
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29
OPS=$R/ops
SPEC=$S/v2/release/specs/dev2-27b-release.json
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
G=/data/dev2/private/panels/goldfree
P=/data/dev2/runs/27b/M3-A-soup/formal/output
HFC=/data/dev2/hf-cache
BASE_REPO=$HFC/models--Qwen--Qwen3.8-27B
BASE=$BASE_REPO/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
FROZEN=/data/dev2/runs/27b/M3-A-soup/formal/triton-cache
FROZEN_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
REPO=llm-semantic-router/DEV2.0-27B
RELEASED=6931828d7e41a5d31cc8e5acdc36f5eebae70fad
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=rename-27b
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton
if [[ -e "$D/DEV2.0-27B.decision.json" ]]; then
  cmp "$R/DEV2.0-27B.decision.json" "$D/DEV2.0-27B.decision.json"
else
  cp "$R/DEV2.0-27B.decision.json" "$D/DEV2.0-27B.decision.json"
fi
if [[ "$preview" == 1 ]]; then
  W=/data/dev2/runs/release/dev2-27b-preview-$TS
  mkdir -p "$W/logs"
  cd "$S" && python3 -m v2.release.build --spec "$SPEC" --output "$W/package/DEV2.0-27B" > "$W/logs/build.log"
  find "$W/package" -name '*.safetensors' -delete
  echo "preview=$W/package/DEV2.0-27B (weights removed after the build)"
  exit 0
fi
if [[ -z "$gpu" ]]; then
  gpu=$(rocm-smi --showuse --showmeminfo vram --json | python3 "$OPS/pick_gpu.py" 120 3 7 6 5 4) \
    || { echo "no node-B GPU (not 0-2) with low use and >= 120 GB free VRAM" >&2; exit 1; }
fi
[[ "$gpu" =~ ^[3-7]$ ]] || { echo "node-B GPU0-2 hold ~27B Milestone 4b; use GPU3-7" >&2; exit 2; }
TC=/data/dev2/runs/release/triton/dev2-27b-f1-copy-$TS
W=/data/dev2/runs/release/dev2-27b-card-$TS
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 1
cd "$S"
python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$FROZEN_SHA" --dest "$TC"
echo "mirror $SRC gpu $gpu cache $TC"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" \
  --image "$IMAGE" --gpu "$gpu" --track release-rename --shared-lease "$LEASE" --threads 4 \
  --site /opt/decision-fla --require-kernels --base-path "$BASE" \
  --env HIP_FORCE_DEV_KERNARG=1 --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" \
  --env "HF_HUB_CACHE=$HFC" --mount "$BASE_REPO" --mount "$HFC/blobs" \
  --mount "$G" --mount "$P" \
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:200" \
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:300" \
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:100" \
  --upload --collect --already-collected || status=$?
set +x
python3 -m v2.27b.triton_cache finish --dest "$TC" || true
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
mkdir -p "$W/extra"
"$HFPY" "$OPS/revision_diff.py" "$REPO" "$RELEASED" "$REV" "$W/extra/revision-diff.json" || status=1
"$HFPY" "$OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-27B" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-27B" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || true
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
