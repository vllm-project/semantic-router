"""Pavlick & Tetreault formality: how formal is the writing style of this sentence?

News, blog and email sentences only (the Yahoo Answers domain is dropped); test first,
then train. Gold: the mean human score on -3..3 cut at -1.8 / -0.6 / 0.6 / 1.8 into five
levels. There is no document id, so each sentence is its own group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, level, make, read_csv, score, spec

TASK = "formality/pavlick"
FILES = (("test.csv", "test"), ("train.csv", "train"))
DOMAINS = {"news", "blog", "email"}
CUTS = (-1.8, -0.6, 0.6, 1.8)
QUESTION = score(
    "The state is one sentence. How formal is the writing style of this sentence?",
    ["very informal", "somewhat informal", "neutral", "somewhat formal", "very formal"],
)

SPEC = spec(
    key="pavlick",
    dataset_id="osyvokon/pavlick-formality-scores",
    revision="904009d6f19d0d4eabb8f1471a92d49c839920d5",
    licence="cc-by-3.0",
    evidence="README.md at the pinned revision (re-upload): card licence cc-by-3.0; the "
    "authors' own licence statement was not found",
    label_provenance="5 MTurk ratings per sentence on -3..3, averaged",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for relative, split in FILES:
        for index, row in enumerate(read_csv(root / relative)):
            sentence = row["sentence"].strip()
            if row["domain"].strip() not in DOMAINS or not sentence:
                continue
            try:
                value = float(row["avg_score"])
            except ValueError:
                continue
            item_id = f"{split}:{index}"
            item = make(
                SPEC,
                TASK,
                item_id,
                item_id,
                split,
                relative,
                index,
                {"sentence": sentence},
                dict(QUESTION),
                level(value, CUTS),
                overlap_texts=[sentence],
            )
            if item:
                yield item
