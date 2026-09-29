#!/usr/bin/env bash
# Finish of the DEV2.0-8B collection add after final-collect.sh stopped in `hub collect`: the gate seal bound the
# final decision to revision 53bac735, but `collect` refused because the collection title is now "🎲 Decision 2.0"
# (renamed outside the pipeline; the builder expects "Decision 2.0"). DEV2.0-8B was already an item, so nothing is
# added here: the post-collect readback runs as release.sh would run it (--expect-collected), then the DEV2.0 items are
# put in size order inside their positions, then card HTTP, link and headroom checks. Node A, no GPU.
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
W=${1:?work dir of final-collect.sh}
HFPY=/data/dev2/tools/hf-cli/bin/python
REPO=llm-semantic-router/DEV2.0-8B
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
cd "$S" && export PYTHONPATH="$S" && mkdir -p "$W/extra"
"$HFPY" -m v2.release.hub readback --repo "$REPO" --revision "$REV" --package "$W/package/DEV2.0-8B" \
  --expect-collected --output "$W/extra/readback-collected.json"
"$HFPY" - "$COLL" "$W/extra/collection-order.json" <<'PY'
import json, re, sys
from huggingface_hub import HfApi
api, slug = HfApi(), sys.argv[1]
def size(item_id):
    m = re.fullmatch(r"llm-semantic-router/DEV2\.0-([0-9.]+)B", item_id)
    return float(m.group(1)) if m else None
before = api.get_collection(slug)
ours = [i for i in before.items if size(i.item_id) is not None]
slots = sorted(i.position for i in ours)
for position, item in zip(slots, sorted(ours, key=lambda i: size(i.item_id))):
    if item.position != position:
        api.update_collection_item(slug, item.item_object_id, position=position)
after = api.get_collection(slug)
out = {"slug": slug, "title": after.title, "private": after.private,
       "before": [[i.position, i.item_id] for i in before.items],
       "after": [[i.position, i.item_id] for i in after.items]}
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False))
PY
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-8B" --output "$W/extra/card-http.json"
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/DEV2.0-8B" --output "$W/extra/hub-links.json"
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV"
