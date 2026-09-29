"""Scruples Anecdotes: according to most readers, is the author in the wrong?

Test, then dev, then train (`anecdotes.tar.gz` v1.0). Gold is `binarized_label`
(WRONG = yes, RIGHT = no). State: the title and the first 2,000 characters of the
story; vote counts, the action and the post type are never shown. The post is the group.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import BODY_CHARS, HtCandidate, make, noul, spec

TASK = "moral/scruples"
ARCHIVE = "anecdotes.tar.gz"
MEMBERS = (
    ("anecdotes/test.scruples-anecdotes.jsonl", "test"),
    ("anecdotes/dev.scruples-anecdotes.jsonl", "validation"),
    ("anecdotes/train.scruples-anecdotes.jsonl", "train"),
)
GOLD = {"WRONG": True, "RIGHT": False}
QUESTION = noul(
    "The state is a story that its author posted to an online community, asking "
    "readers to judge who was in the wrong: its title and the beginning of the story.",
    "According to most readers, the author of this story is in the wrong.",
    "According to most readers, the author of this story is not in the wrong.",
)

SPEC = spec(
    key="scruples",
    dataset_id="allenai/scruples (Anecdotes v1.0)",
    revision="c79697e97d24f5b43b5ff030a9b16011a32e4a5f",
    licence="apache-2.0",
    evidence="LICENSE at the pinned commit (Apache-2.0); archive from the readme link",
    label_provenance="community verdict votes on the posts (majority, binarized)",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    with tarfile.open(root / ARCHIVE) as archive:
        for member, split in MEMBERS:
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                gold = GOLD.get(row.get("binarized_label"))
                title = " ".join(str(row.get("title") or "").split())
                story = str(row.get("text") or "").strip()[:BODY_CHARS]
                if gold is None or not story:
                    continue
                item = make(
                    SPEC,
                    TASK,
                    str(row["id"]),
                    str(row.get("post_id") or row["id"]),
                    split,
                    f"{ARCHIVE}:{member}",
                    index,
                    {"title": title, "story": story},
                    dict(QUESTION),
                    gold,
                    overlap_texts=[title, story],
                )
                if item:
                    yield item
