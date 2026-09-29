"""Keep the DEV2.0 items of the pinned collection in size order (hf-cli python, node token).

  <hf-cli python> collection_order.py <slug> <out.json>

The DEV2.0 items are sorted by size inside the positions they already occupy; other items, the
title and the privacy are left alone. Exits 1 unless the result is exactly the six expected
releases in order and the collection is private.
"""

from __future__ import annotations

import json
import re
import sys

from huggingface_hub import HfApi

EXPECTED = [
    f"llm-semantic-router/DEV2.0-{s}" for s in ("0.6B", "0.8B", "2B", "4B", "9B", "27B")
]


def size(item_id: str) -> float | None:
    m = re.fullmatch(r"llm-semantic-router/DEV2\.0-([0-9.]+)B", item_id)
    return float(m.group(1)) if m else None


def main() -> int:
    slug, out = sys.argv[1:3]
    api = HfApi()
    before = api.get_collection(slug)
    ours = [i for i in before.items if size(i.item_id) is not None]
    slots = sorted(i.position for i in ours)
    moved = []
    for position, item in zip(slots, sorted(ours, key=lambda i: size(i.item_id))):
        if item.position != position:
            api.update_collection_item(slug, item.item_object_id, position=position)
            moved.append([item.item_id, item.position, position])
    after = api.get_collection(slug)
    releases = [
        i.item_id
        for i in sorted(after.items, key=lambda i: i.position)
        if size(i.item_id) is not None
    ]
    receipt = {
        "slug": slug,
        "title": after.title,
        "title_unchanged": after.title == before.title,
        "private": after.private,
        "before": [[i.position, i.item_id, i.item_object_id] for i in before.items],
        "moved": moved,
        "after": [[i.position, i.item_id, i.item_object_id] for i in after.items],
        "releases_in_order": releases == EXPECTED,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("title_unchanged", "private", "moved", "releases_in_order")
            }
        )
    )
    return (
        0
        if receipt["releases_in_order"]
        and after.private is True
        and receipt["title_unchanged"]
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
