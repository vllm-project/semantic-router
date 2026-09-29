"""IBM claim stance: does the claim argue in favour of the topic?

Test topics first, then train. PRO = yes, CON = no. The topic is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, noul, read_csv, spec

TASK = "stance/claim_stance"
FILES = (("test.csv", "test"), ("train.csv", "train"))
GOLD = {"PRO": True, "CON": False}
QUESTION = noul(
    "The state gives a debate topic and a claim taken from an encyclopedia article.",
    "The claim argues in favour of the topic.",
    "The claim argues against the topic.",
)

SPEC = spec(
    key="claim_stance",
    dataset_id="ibm-research/claim_stance",
    revision="ec4e2c2ec3e0c70087c67a28a7bce58b682b8109",
    licence="cc-by-sa-3.0",
    evidence="README.md at the pinned revision: card licence cc-by-3.0; body '(c) "
    "Copyright IBM 2014. Released under CC-BY-SA 3.0'",
    label_provenance="IBM expert annotators labelled each claim PRO/CON vs the topic",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for relative, split in FILES:
        for index, row in enumerate(read_csv(root / relative)):
            gold = GOLD.get(row["claims.stance"].strip())
            topic = row["topicText"].strip()
            claim = row["claims.claimCorrectedText"].strip()
            if gold is None or not topic or not claim:
                continue
            item = make(
                SPEC,
                TASK,
                f"{split}:{row['claims.claimId']}",
                row["topicId"],
                split,
                relative,
                index,
                {"topic": topic, "claim": claim},
                dict(QUESTION),
                gold,
                overlap_texts=[claim],
            )
            if item:
                yield item
