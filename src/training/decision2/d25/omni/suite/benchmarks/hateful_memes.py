"""Moderation (Hateful Memes): dev_seen + test_seen rows whose image is in the Hub copy, 1,213 rows.

``neuralcatcher/hateful_memes`` holds 9,664 of the dataset's images; dev_seen keeps 398 of 500 rows
and test_seen 815 of 1,000, which is the board's count exactly. Binary choice (hateful / not
hateful) on the meme image. The caption-only control is the ``hateful-memes-caption-only`` variant
(same question, caption text as the state, no image); the other splits are variants too.
"""

from __future__ import annotations

import json
from pathlib import Path

from d25.omni.suite import rows as R

BENCHMARK = "Moderation (Hateful Memes)"
SOURCES = ("hateful_memes",)
BOARD_SPLITS = ("dev_seen", "test_seen")
OTHER_SPLITS = ("dev_unseen", "test_unseen")
INSTRUCTIONS = "Is this meme hateful? Judge the picture and its caption together."
CAPTION_INSTRUCTIONS = "Is this meme hateful? Judge its caption."
CRITERIA = {
    "hateful": "The meme is hateful.",
    "not_hateful": "The meme is not hateful.",
}


def _rows(ctx, split: str):
    root = ctx.source("hateful_memes")
    name = f"{split}.jsonl"
    for line in (root / name).read_text().splitlines():
        if line.strip():
            yield name, json.loads(line)


def build(ctx) -> dict:
    root = ctx.source("hateful_memes")
    board, caption, others, missing = [], [], [], {}
    for split in BOARD_SPLITS + OTHER_SPLITS:
        for name, r in _rows(ctx, split):
            path = root / r["img"]
            if not path.exists():
                missing[split] = missing.get(split, 0) + 1
                continue
            gold = "hateful" if int(r["label"]) == 1 else "not_hateful"
            prov = ctx.provenance("hateful_memes", name, r["id"], image_file=r["img"])
            common = dict(
                benchmark=BENCHMARK,
                split=split,
                source_id=str(r["id"]),
                criteria=CRITERIA,
                gold=gold,
                provenance=prov,
            )
            row = R.make_row(
                images=[ctx.store(Path(path).read_bytes())],
                instructions=INSTRUCTIONS,
                **common,
            )
            (board if split in BOARD_SPLITS else others).append(row)
            if split in BOARD_SPLITS:
                caption.append(
                    R.make_row(
                        images=[],
                        instructions=CAPTION_INSTRUCTIONS,
                        state={"caption": r["text"]},
                        tags=["caption-only"],
                        **common,
                    )
                )
    return {
        "rows": board,
        "variants": {
            "hateful-memes-caption-only": caption,
            "hateful-memes-unseen": others,
        },
        "notes": f"dev_seen + test_seen with an image in the Hub copy; rows without an image: {missing}",
    }
