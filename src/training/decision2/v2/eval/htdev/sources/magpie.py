"""MAGPIE (backup for figurative): is the expression used figuratively in this sentence?

One released file (`all`). Gold: `usage` figurative = yes, literal = no (other values
dropped). The idiom is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, noul, read_csv, spec

TASK = "figurative/magpie"
FILE = "magpie.tsv"
GOLD = {"figurative": True, "literal": False}
QUESTION = noul(
    "The state gives a sentence and a potentially idiomatic expression that occurs in it.",
    "The expression is used figuratively in this sentence.",
    "The expression is used literally in this sentence.",
)

SPEC = spec(
    key="magpie",
    dataset_id="gsarti/magpie",
    revision="fa6ae9d93b03e6403e82696496dfbd2cf5c3d3d5",
    licence="cc-by-4.0",
    evidence="github.com/hslh/magpie-corpus LICENSE @7fa677b8 (CC BY 4.0); mirror card "
    "cc-by-4.0",
    label_provenance="crowd annotators (at least 2, with confidence), partly adjudicated",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for index, row in enumerate(read_csv(root / FILE, "\t")):
        gold = GOLD.get(row["usage"].strip())
        sentence = row["sentence"].strip()
        idiom = row["idiom"].strip()
        if gold is None or not sentence or not idiom:
            continue
        item = make(
            SPEC,
            TASK,
            f"row{index}",
            idiom.casefold(),
            "all",
            FILE,
            index,
            {"sentence": sentence, "expression": idiom},
            dict(QUESTION),
            gold,
            overlap_texts=[sentence],
        )
        if item:
            yield item
