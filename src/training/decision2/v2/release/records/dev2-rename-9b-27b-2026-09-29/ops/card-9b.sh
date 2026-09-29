#!/usr/bin/env bash
# DEV2.0-9B card-only revision after the move from DEV2.0-8B (user directive 2026-09-29 16:05 UTC+8): release.sh
# --upload --collect --already-collected from this mirror (spec dev2-9b-release.json, final decision
# DEV2.0-9B.decision.json), subset parity on the four scored panels with a fresh copy of the scored run's autotune
# cache, on one node-A GPU under the shared lease owner.rename-9b. GPU6-7 hold 9B round-1 training and are never used.
# Afterwards: every weight file byte-identical to the released 53bac735, collection order, card HTTP, links, storage.
# Usage (node A): bash <mirror>/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops/card-9b.sh [--preview] [--gpu N]
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
SPEC=$S/v2/release/specs/dev2-9b-release.json
G=/data/dev2/private/panels/goldfree
P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived
FROZEN=/data/dev2/runs/9b/formal-m4/triton-cache
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
REPO=llm-semantic-router/DEV2.0-9B
RELEASED=53bac735be58def53673d0d290b9baa3f2af1cf9
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=rename-9b
export PYTHONPATH=$S
if [[ -e "$D/DEV2.0-9B.decision.json" ]]; then
  cmp "$R/DEV2.0-9B.decision.json" "$D/DEV2.0-9B.decision.json"
else
  cp "$R/DEV2.0-9B.decision.json" "$D/DEV2.0-9B.decision.json"
fi
if [[ "$preview" == 1 ]]; then
  W=/data/dev2/runs/release/dev2-9b-preview-$TS
  mkdir -p "$W/logs"
  cd "$S" && python3 -m v2.release.build --spec "$SPEC" --output "$W/package/DEV2.0-9B" > "$W/logs/build.log"
  find "$W/package" -name '*.safetensors' -delete
  echo "preview=$W/package/DEV2.0-9B (weights removed after the build)"
  exit 0
fi
if [[ -z "$gpu" ]]; then
  gpu=$(rocm-smi --showuse --showmeminfo vram --json | python3 "$OPS/pick_gpu.py" 60 1 0 5 4 3 2) \
    || { echo "no node-A GPU (not 6-7) with low use and >= 60 GB free VRAM" >&2; exit 1; }
fi
[[ "$gpu" =~ ^[0-5]$ ]] || { echo "node-A GPU6-7 hold 9B round-1 training; use GPU0-5" >&2; exit 2; }
TC=/data/dev2/runs/release/triton/dev2-9b-K-a13-copy-$TS
W=/data/dev2/runs/release/dev2-9b-card-$TS
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 1
cp -a "$FROZEN" "$TC"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
echo "mirror $SRC gpu $gpu; cache copy $TC files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$FROZEN")"
set -x
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" \
  --gpu "$gpu" --track release-rename --shared-lease "$LEASE" --threads 4 \
  --site /opt/decision-fla --require-kernels \
  --env HIP_FORCE_DEV_KERNARG=1 --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR="$TC" --mount-rw "$TC" \
  --mount "$G" --mount "$P" \
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:200" \
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:300" \
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:100" \
  --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:100" \
  --upload --collect --already-collected
set +x
echo "cache after run files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$FROZEN")"
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
mkdir -p "$W/extra"
cd "$S"
status=0
"$HFPY" "$OPS/revision_diff.py" "$REPO" "$RELEASED" "$REV" "$W/extra/revision-diff.json" || status=1
"$HFPY" "$OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-9B" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-9B" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || true
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
