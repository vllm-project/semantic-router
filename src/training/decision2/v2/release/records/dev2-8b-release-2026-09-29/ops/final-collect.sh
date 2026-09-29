#!/usr/bin/env bash
# DEV2.0-8B collection add (coordinator decision 2026-09-29 15:49 UTC+8): release.sh --upload --collect from the mirror
# whose spec gate_receipt names the final decision, with subset parity on the four scored panels and a fresh copy of the
# scored run's autotune cache. DEV2.0-8B was already an item of the collection before this seal (added outside the
# pipeline), so the readbacks run with --already-collected. Afterwards the DEV2.0 items are put in size order inside the
# positions they occupy (other items keep theirs), then readback, card HTTP and link checks. Node A GPU6.
set -euo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-8b-release-2026-09-29
G=/data/dev2/private/panels/goldfree
P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived
FROZEN=/data/dev2/runs/9b/formal-m4/triton-cache
TC=/data/dev2/runs/release/triton/dev2-8b-K-a13-copy-$TS
W=/data/dev2/runs/release/dev2-8b-final-$TS
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
REPO=llm-semantic-router/DEV2.0-8B
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
trap 'rm -f /data/dev2/leases/gpu6.lock/owner.release' EXIT
if [[ -e "$D/DEV2.0-8B.decision.json" ]]; then
  cmp "$R/DEV2.0-8B.decision.json" "$D/DEV2.0-8B.decision.json"
else
  cp "$R/DEV2.0-8B.decision.json" "$D/DEV2.0-8B.decision.json"
fi
# storage guard (the weights are already in the repository; README and MODEL_MANIFEST may change)
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 1
cp -a "$FROZEN" "$TC"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
echo "mirror $SRC; cache copy $TC files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$FROZEN")"
set -x
"$S/v2/release/release.sh" --spec "$S/v2/release/specs/dev2-8b-release.json" --src "$SRC" --work "$W" \
  --gpu 6 --track release-9b --shared-lease release --threads 4 \
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
"$HFPY" - "$COLL" "$W/extra/collection-order.json" <<'PY'
import json, re, sys
from huggingface_hub import HfApi
api, slug = HfApi(), sys.argv[1]
def size(item_id):
    m = re.fullmatch(r"llm-semantic-router/DEV2\.0-([0-9.]+)B", item_id)
    return float(m.group(1)) if m else None
before = api.get_collection(slug).items
ours = [i for i in before if size(i.item_id) is not None]
slots = sorted(i.position for i in ours)
for position, item in zip(slots, sorted(ours, key=lambda i: size(i.item_id))):
    if item.position != position:
        api.update_collection_item(slug, item.item_object_id, position=position)
after = api.get_collection(slug)
out = {"slug": slug, "private": after.private,
       "before": [[i.position, i.item_id] for i in before],
       "after": [[i.position, i.item_id] for i in after.items]}
json.dump(out, open(sys.argv[2], "w"), indent=2)
print(json.dumps(out))
PY
cd "$S" && export PYTHONPATH="$S"
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-8B" --output "$W/extra/card-http.json"
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-8B" --output "$W/extra/hub-links.json"
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV"
