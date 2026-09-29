"""Fig-QA: which interpretation expresses what the figurative statement means?

Validation (`dev.csv`) first, then train; the test labels are withheld (-1) and never
used. Paired items share their two endings with opposite meanings, so the unordered
ending pair is the group (one item per pair by default).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, pairwise, read_csv, spec
from v2.eval.sealed.schema import normalized

TASK = "figurative/figqa"
FILES = (("dev.csv", "validation"), ("train.csv", "train"))

SPEC = spec(
    key="figqa",
    dataset_id="nightingal3/fig-qa",
    revision="b29ec2faf6ef73d634db9757f8741dee68f6c874",
    licence="mit",
    evidence="README.md at the pinned revision: licence mit (MIT License text)",
    label_provenance="crowd-authored (MTurk) paired figurative statements with their "
    "correct interpretations, validated by other workers",
    tasks=(TASK,),
)

QUESTION = {
    "type": "choice",
    "instructions": (
        "The state holds a figurative statement and two possible interpretations, "
        "Interpretation A and Interpretation B. Which interpretation expresses what the "
        "figurative statement means?"
    ),
    "criteria": {
        "A": "Interpretation A expresses what the statement means.",
        "B": "Interpretation B expresses what the statement means.",
    },
}


def candidates(root: Path) -> Iterator[HtCandidate]:
    for relative, split in FILES:
        for index, row in enumerate(read_csv(root / relative)):
            statement = row["startphrase"].strip()
            endings = [row["ending1"].strip(), row["ending2"].strip()]
            if row["labels"] not in ("0", "1") or not statement or not all(endings):
                continue
            item_id = f"{split}:{index}"
            shown, _, gold = pairwise(item_id, endings, int(row["labels"]))
            group = "|".join(sorted(normalized(e) for e in endings))
            state = {
                "statement": statement,
                "interpretation_a": shown[0],
                "interpretation_b": shown[1],
            }
            item = make(
                SPEC,
                TASK,
                item_id,
                group,
                split,
                relative,
                index,
                state,
                dict(QUESTION),
                gold,
                overlap_texts=[statement, *endings],
                option_texts=shown,
            )
            if item:
                yield item
