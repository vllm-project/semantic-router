"""RumourEval-2019 (backup for stance): what stance does the reply take?

The pinned HF release ships subtask A only (source text, reply text, stance label);
the subtask B veracity labels are not in it, so this module has no veracity task.
Test, then validation, then train. The source post (thread) is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, choice, make, read_csv, spec

TASK = "stance/rumoureval"
FILES = (
    ("rumoureval2019_test.csv", "test"),
    ("rumoureval2019_val.csv", "validation"),
    ("rumoureval2019_train.csv", "train"),
)
OPTIONS = [
    ("support", "It supports the original post's claim."),
    ("deny", "It denies the original post's claim."),
    ("query", "It asks a question about the original post's claim."),
    ("comment", "It comments without taking a stance on the claim."),
]
INSTRUCTIONS = (
    "The state gives a social-media post that spreads a claim and a reply to it. What "
    "stance does the reply take toward the original post's claim?"
)

SPEC = spec(
    key="rumoureval",
    dataset_id="strombergnlp/rumoureval_2019",
    revision="c9c0c7279d591d2fa4d692501d85f4e46d4b0572",
    licence="cc-by-4.0",
    evidence="README.md at the pinned revision: 'The authors distribute this data under "
    "... CC-BY 4.0'",
    label_provenance="annotated reply stance toward the rumour source post",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    keys = {key for key, _ in OPTIONS}
    for relative, split in FILES:
        for index, row in enumerate(read_csv(root / relative)):
            gold = (row.get("label") or "").strip()
            source = (row.get("source_text") or "").strip()
            reply = (row.get("reply_text") or "").strip()
            if gold not in keys or not source or not reply:
                continue
            item_id = f"{split}:{row['id']}"
            item = make(
                SPEC,
                TASK,
                item_id,
                " ".join(source.split()),
                split,
                relative,
                index,
                {"original_post": source, "reply": reply},
                choice(item_id, INSTRUCTIONS, OPTIONS),
                gold,
                overlap_texts=[source, reply],
            )
            if item:
                yield item
