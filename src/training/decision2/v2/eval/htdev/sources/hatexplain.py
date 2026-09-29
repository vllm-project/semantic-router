"""HateXplain (backup for hate): how would most annotators classify this post?

`dataset.json` posts with three annotator labels; gold is the majority label (posts
without a majority are dropped). Rationales and targets are never read. Splits come
from `post_id_divisions.json`. The post is the group.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, choice, make, spec

TASK = "hate/hatexplain"
FILE = "Data/dataset.json"
DIVISIONS = "Data/post_id_divisions.json"
SPLITS = {"test": "test", "val": "validation", "train": "train"}
GOLD = {"hatespeech": "hate_speech", "offensive": "offensive", "normal": "normal"}
OPTIONS = [
    ("hate_speech", "Hate speech"),
    ("offensive", "Offensive but not hate speech"),
    ("normal", "Normal"),
]
INSTRUCTIONS = (
    "The state is a social-media post. How would most annotators classify this post?"
)

SPEC = spec(
    key="hatexplain",
    dataset_id="punyajoy/HateXplain",
    revision="01d742279dac941981f53806154481c0e15ee686",
    licence="mit",
    evidence="LICENSE at the pinned commit: MIT License (HF card of the authors' "
    "loader: cc-by-4.0)",
    label_provenance="3 MTurk annotators per post; majority label",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    divisions = json.loads((root / DIVISIONS).read_text(encoding="utf-8"))
    split_of = {
        post: SPLITS[name] for name, posts in divisions.items() for post in posts
    }
    posts = json.loads((root / FILE).read_text(encoding="utf-8"))
    for index, (post_id, post) in enumerate(posts.items()):
        counts = Counter(a.get("label") for a in post.get("annotators") or [])
        label, top = counts.most_common(1)[0] if counts else (None, 0)
        gold = GOLD.get(label)
        text = " ".join(post.get("post_tokens") or []).strip()
        if gold is None or 2 * top <= sum(counts.values()) or not text:
            continue
        item = make(
            SPEC,
            TASK,
            post_id,
            post_id,
            split_of.get(post_id, "all"),
            FILE,
            index,
            {"post": text},
            choice(post_id, INSTRUCTIONS, OPTIONS),
            gold,
            overlap_texts=[text],
        )
        if item:
            yield item
