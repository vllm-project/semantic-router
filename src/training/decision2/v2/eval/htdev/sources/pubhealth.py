"""PUBHEALTH: what verdict would professional fact-checkers give this health claim?

The HF parquet conversion of the authors' loader (label names false / mixture / true /
unproven). `unproven` and unlabelled rows are dropped. Only the claim is shown: the
explanation, main text, sources, fact-checker names, dates and subjects never are.
The claim id is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, choice, make, read_parquet, spec

TASK = "misinfo/pubhealth"
FILES = (
    ("default/test/0000.parquet", "test"),
    ("default/validation/0000.parquet", "validation"),
    ("default/train/0000.parquet", "train"),
)
GOLD = {0: "false", 1: "mixture", 2: "true"}
OPTIONS = [("true", "True"), ("false", "False"), ("mixture", "Mixture (partly true)")]
INSTRUCTIONS = (
    "The state is a public-health claim that circulated online or in the news. What "
    "verdict would professional fact-checkers give this health claim?"
)

SPEC = spec(
    key="pubhealth",
    dataset_id="ImperialCollegeLondon/health_fact (refs/convert/parquet)",
    revision="7f898a16838e708a4597986e3fab8af05b710ba4",
    licence="mit",
    evidence="github.com/neemakot/Health-Fact-Checking LICENSE @02136d3e (MIT, Neema "
    "Kotonya); HF card of the authors' loader @57995242: licence mit",
    label_provenance="verdicts of professional fact-checking and health-news review "
    "sites, normalised to true / false / mixture / unproven",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for relative, split in FILES:
        rows = read_parquet(root / relative, ["claim_id", "claim", "label"])
        for index, row in enumerate(rows):
            gold = GOLD.get(row["label"])
            claim = " ".join(str(row["claim"] or "").split())
            if gold is None or not claim:
                continue
            item_id = f"{split}:{row['claim_id']}"
            item = make(
                SPEC,
                TASK,
                item_id,
                str(row["claim_id"]),
                split,
                relative,
                index,
                {"claim": claim},
                choice(item_id, INSTRUCTIONS, OPTIONS),
                gold,
                overlap_texts=[claim],
            )
            if item:
                yield item
