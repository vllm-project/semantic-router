"""Circa: how would X most likely interpret Y's indirect answer?

Single released split (`train`). Gold is the relaxed majority label `goldstandard2`;
`Other` (and any unlabelled row) is dropped. The crowd `judgements` are never read into
the state. The group is the (situation, question) pair, since many answers share one.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, choice, make, read_parquet, spec

TASK = "implicature/circa"
FILE = "data/train-00000-of-00001.parquet"
GOLD = {0: "yes", 1: "no", 2: "in_the_middle", 3: "yes_conditional"}
OPTIONS = [
    ("yes", "Yes"),
    ("no", "No"),
    ("yes_conditional", "Yes, subject to some conditions"),
    ("in_the_middle", "In the middle, neither yes nor no"),
]
INSTRUCTIONS = (
    "The state gives a situation, a yes/no question that X asked Y, and Y's answer, "
    "which does not say yes or no directly. How would X most likely interpret Y's answer?"
)

SPEC = spec(
    key="circa",
    dataset_id="google-research-datasets/circa",
    revision="faa1b5a78dd926a899bcd4da289c2e3abe8061a9",
    licence="cc-by-4.0",
    evidence="README.md at the pinned revision: licence cc-by-4.0 (CC Attribution 4.0)",
    label_provenance="5 crowd annotators per (question, answer); goldstandard2 = relaxed "
    "majority label",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    columns = ["context", "question-X", "answer-Y", "goldstandard2"]
    for index, row in enumerate(read_parquet(root / FILE, columns)):
        gold = GOLD.get(row["goldstandard2"])
        situation = str(row["context"] or "").strip()
        question = str(row["question-X"] or "").strip()
        answer = str(row["answer-Y"] or "").strip()
        if gold is None or not (situation and question and answer):
            continue
        item_id = f"row{index}"
        state = {"situation": situation, "x_question": question, "y_answer": answer}
        item = make(
            SPEC,
            TASK,
            item_id,
            f"{situation}|{question}",
            "train",
            FILE,
            index,
            state,
            choice(item_id, INSTRUCTIONS, OPTIONS),
            gold,
            overlap_texts=[question, answer],
        )
        if item:
            yield item
