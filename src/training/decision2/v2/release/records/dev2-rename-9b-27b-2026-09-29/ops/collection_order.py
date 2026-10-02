"""Read back the pinned collection after a release (hf-cli python, node token).

  <hf-cli python> collection_order.py <slug> <out.json>

The collection's order and title are curated by hand, so this script never changes them. Exits 1
unless the collection is private and its models are exactly the six expected releases, in any
order. Receipt fields keep their earlier names; ``moved`` is always empty.
"""

from __future__ import annotations

import json
import sys

from huggingface_hub import HfApi

EXPECTED = [
    f"llm-semantic-router/Decision-2.0-{name}"
    for name in ("Kai-0.6B", "Eos-0.8B", "Sol-2B", "Nox-4B", "Lux-9B", "Vega-27B")
]


def main() -> int:
    slug, out = sys.argv[1:3]
    collection = HfApi().get_collection(slug)
    items = sorted(collection.items, key=lambda i: i.position)
    models = [i.item_id for i in items if i.item_type == "model"]
    releases_present = sorted(models) == sorted(EXPECTED)
    receipt = {
        "slug": slug,
        "title": collection.title,
        "title_unchanged": True,
        "private": collection.private,
        "before": [[i.position, i.item_id, i.item_object_id] for i in items],
        "moved": [],
        "after": [[i.position, i.item_id, i.item_object_id] for i in items],
        "expected": EXPECTED,
        "releases_in_order": releases_present,
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
    return 0 if releases_present and collection.private is True else 1


if __name__ == "__main__":
    sys.exit(main())
