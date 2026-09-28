"""Empathic reactions to news: how much empathic concern did the writer feel?

One released file (`all`). Gold: the writer's own empathy score (Batson scale, 1-7)
cut at 2.2 / 3.4 / 4.6 / 5.8 into five levels; distress and the bins are never shown.
The writer (`response_id`, five messages each) is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, level, make, read_csv, score, spec

TASK = "empathy/empathic_reactions"
FILE = "data/responses/data/messages.csv"
CUTS = (2.2, 3.4, 4.6, 5.8)
QUESTION = score(
    "The state is a message that a writer wrote right after reading a news story. How "
    "much empathic concern did the writer feel for the people in the story?",
    ["none", "little", "moderate", "strong", "very strong"],
)

SPEC = spec(
    key="empathic_reactions",
    dataset_id="wwbp/empathic_reactions",
    revision="3ddbf6b35c789f99c930a349f3fc26b4b2d9ae2e",
    licence="cc-by-4.0",
    evidence="README.md at the pinned commit: 'Our dataset is available under CC BY 4.0'",
    label_provenance="writers' own empathic-concern self-report (Batson items, 1-7 mean)",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for index, row in enumerate(read_csv(root / FILE)):
        essay = row["essay"].strip()
        try:
            value = float(row["empathy"])
        except ValueError:
            continue
        if not essay:
            continue
        item = make(
            SPEC,
            TASK,
            row["message_id"],
            row["response_id"],
            "all",
            FILE,
            index,
            {"message": essay},
            dict(QUESTION),
            level(value, CUTS),
            overlap_texts=[essay],
        )
        if item:
            yield item
