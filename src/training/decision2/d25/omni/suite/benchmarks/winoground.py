"""Winoground: 400 examples x 2 images = 800 rows; each row is one image and the two captions.

Caption order is the source order (caption_0 = A, caption_1 = B), so image_0 rows have gold A and
image_1 rows gold B. Gated dataset under the Meta Images Research License: evaluation only, never
redistributed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from d25.omni.suite import rows as R

BENCHMARK = "Winoground"
SOURCES = ("winoground",)
INSTRUCTIONS = "Which caption describes this image?"


def build(ctx) -> dict:
    root = ctx.source("winoground") / "data"
    out = []
    with zipfile.ZipFile(root / "images.zip") as z:
        members = {Path(n).stem: n for n in z.namelist() if not n.endswith("/")}
        for line in (root / "examples.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for i in (0, 1):
                image = r[f"image_{i}"]
                source_id = f"{r['id']}-{i}"
                out.append(
                    R.make_row(
                        benchmark=BENCHMARK,
                        split="test",
                        source_id=source_id,
                        images=[ctx.store(z.read(members[image]))],
                        instructions=INSTRUCTIONS,
                        criteria={"A": r["caption_0"], "B": r["caption_1"]},
                        gold="AB"[i],
                        provenance=ctx.provenance(
                            "winoground",
                            "data/examples.jsonl",
                            source_id,
                            image_member=members[image],
                        ),
                        tags=[
                            f"tag:{r.get('tag')}",
                            f"collapsed:{r.get('collapsed_tag')}",
                        ],
                        extra={"example_id": r["id"], "image_index": i},
                    )
                )
    return {
        "rows": out,
        "variants": {},
        "notes": "one image + both captions per row, source caption order",
    }
