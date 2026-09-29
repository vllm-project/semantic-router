"""MFRC (backup for moral): does the comment express a moral judgement or concern?

Annotation rows are grouped per comment text. Gold: yes when a strict majority of its
annotators gave any moral label (a foundation or Thin Morality), no when a strict
majority gave Non-Moral; ties are dropped. Subreddit, bucket and annotator fields are
never shown. Single split (`all`); the comment is the group.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, noul, read_csv, spec

TASK = "moral/mfrc"
FILE = "final_mfrc_data.csv"
QUESTION = noul(
    "The state is a comment posted on an online discussion forum.",
    "The comment expresses a moral judgement or moral concern (for example about care, "
    "fairness, loyalty, authority or purity).",
    "The comment expresses no moral judgement or moral concern.",
)

SPEC = spec(
    key="mfrc",
    dataset_id="USC-MOLA-Lab/MFRC",
    revision="ddc21d2f03e156732fd1ba95a51c4c29f07975be",
    licence="cc-by-4.0",
    evidence="README.md at the pinned revision: 'Licensing Information: cc-by-4.0'",
    label_provenance="at least 3 trained annotators per comment; moral foundations or "
    "Non-Moral",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    votes: dict[str, list[bool]] = defaultdict(list)
    first: dict[str, int] = {}
    for index, row in enumerate(read_csv(root / FILE)):
        text = row["text"].strip()
        labels = {v.strip() for v in row["annotation"].split(",") if v.strip()}
        if not text or not labels:
            continue
        first.setdefault(text, index)
        votes[text].append(labels != {"Non-Moral"})
    for text, marks in votes.items():
        moral = sum(marks)
        if 2 * moral == len(marks):
            continue
        index = first[text]
        item = make(
            SPEC,
            TASK,
            f"row{index}",
            f"row{index}",
            "all",
            FILE,
            index,
            {"comment": text},
            dict(QUESTION),
            2 * moral > len(marks),
            overlap_texts=[text],
        )
        if item:
            yield item
