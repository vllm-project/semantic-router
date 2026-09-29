#!/usr/bin/env bash
# DEV2.0-8B private release run on node A GPU6: build, System One examples, repeatability, card, full parity on the
# three scored JevArena / JevBench panels and mlx-diag (one fresh copy of the persisted autotune cache that the scored
# run and its mlx-diag run shared), upload, real download, re-hash, post-download checks and Hub readback, then the card
# HTTP and link checks and the gate evaluation. Stops before the collection add (no --collect).
set -euo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-8b-release-2026-09-29
G=/data/dev2/private/panels/goldfree
P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived
FROZEN=/data/dev2/runs/9b/formal-m4/triton-cache
TC=/data/dev2/runs/release/triton/dev2-8b-K-a13-copy-$TS
W=/data/dev2/runs/release/dev2-8b-release-$TS
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
REPO=llm-semantic-router/DEV2.0-8B
trap 'rm -f /data/dev2/leases/gpu6.lock/owner.release' EXIT
mkdir -p "$D"
if [[ -e "$D/DEV2.0-8B.decision.build-draft.json" ]]; then
  cmp "$R/DEV2.0-8B.decision.build-draft.json" "$D/DEV2.0-8B.decision.build-draft.json"
else
  cp "$R/DEV2.0-8B.decision.build-draft.json" "$D/DEV2.0-8B.decision.build-draft.json"
fi
# storage guard right before the run (package 17.97 GB; exit 1 = would not fit)
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 19
cp -a "$FROZEN" "$TC"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
echo "mirror $SRC; cache copy $TC files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$FROZEN")"
set -x
"$S/v2/release/release.sh" --spec "$S/v2/release/specs/dev2-8b-release.json" --src "$SRC" --work "$W" \
  --gpu 6 --track release-9b --shared-lease release --threads 4 \
  --site /opt/decision-fla --require-kernels \
  --env HIP_FORCE_DEV_KERNARG=1 --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR="$TC" --mount-rw "$TC" \
  --mount "$G" --mount "$P" \
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600" \
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547" \
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231" \
  --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275" \
  --upload
set +x
echo "cache after run files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$FROZEN")"
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
cd "$S" && export PYTHONPATH="$S" && mkdir -p "$W/extra"
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-8B" --output "$W/extra/card-http.json"
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-8B" --output "$W/extra/hub-links.json"
python3 -B -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || echo "gate evaluate did not pass: $W/extra/gate-evaluate.json"
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV"
