"""New Yorker Caption Contest ranking pairs: which caption would readers find funnier?

The five ranking cross-validation folds (`ranking`, `ranking_1` .. `ranking_4`) have
test sets over disjoint contests; their union is the pool (split `test`). The gold is
the source ranking label (the official or crowd-voted winner vs a low-rated entry).
State: the human-written cartoon description fields and the two captions in display
order. Entities are Wikipedia URLs in the source; only their page titles are shown.
The contest is the group.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, pairwise, read_parquet, spec

TASK = "humor/nycc_pairs"
FOLDS = ("ranking", "ranking_1", "ranking_2", "ranking_3", "ranking_4")
COLUMNS = [
    "contest_number",
    "image_location",
    "image_description",
    "image_uncanny_description",
    "entities",
    "caption_choices",
    "label",
    "instance_id",
]

SPEC = spec(
    key="nycc",
    dataset_id="jmhessel/newyorker_caption_contest",
    revision="d81cbab7d0392708d5371d3a4960e69261824db4",
    licence="cc-by-4.0",
    evidence="README.md at the pinned revision: licence cc-by-4.0 (annotations CC-BY)",
    label_provenance="official New Yorker finalists/winners or crowd top-rated captions "
    "(reader funniness votes) vs a low-rated entry for the same cartoon",
    tasks=(TASK,),
)

QUESTION = {
    "type": "choice",
    "instructions": (
        "The state describes a New Yorker cartoon (its scene, a description, what is "
        "unusual about it and the entities it shows) and two captions entered in its "
        "caption contest, Caption A and Caption B. Which caption would New Yorker "
        "readers find funnier for this cartoon?"
    ),
    "criteria": {
        "A": "Caption A is the funnier caption for this cartoon.",
        "B": "Caption B is the funnier caption for this cartoon.",
    },
}


def entity_name(url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return urllib.parse.unquote(tail).replace("_", " ").strip()


def candidates(root: Path) -> Iterator[HtCandidate]:
    seen: set[str] = set()
    for fold in FOLDS:
        relative = f"{fold}/test-00000-of-00001.parquet"
        for index, row in enumerate(read_parquet(root / relative, COLUMNS)):
            item_id = str(row["instance_id"])
            captions = [str(c).strip() for c in row["caption_choices"] or []]
            if item_id in seen or len(captions) != 2 or row["label"] not in ("A", "B"):
                continue
            seen.add(item_id)
            shown, _, gold = pairwise(item_id, captions, "AB".index(row["label"]))
            state = {
                "scene": str(row["image_location"] or "").strip(),
                "description": str(row["image_description"] or "").strip(),
                "uncanny": str(row["image_uncanny_description"] or "").strip(),
                "entities": [entity_name(e) for e in row["entities"] or []],
                "caption_a": shown[0],
                "caption_b": shown[1],
            }
            item = make(
                SPEC,
                TASK,
                item_id,
                str(row["contest_number"]),
                "test",
                relative,
                index,
                state,
                dict(QUESTION),
                gold,
                overlap_texts=[state["description"], state["uncanny"], *captions],
                option_texts=shown,
            )
            if item:
                yield item
