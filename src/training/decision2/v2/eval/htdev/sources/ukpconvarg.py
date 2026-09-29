"""UKPConvArg1Strict: which argument is more convincing for this stance?

One tab-separated file per topic-stance (32; no split, `all`). The topic and stance are
the file name's two parts (split at the first `_`, hyphens read as spaces). Gold is the
crowd majority (MACE, strict agreement) `a1`/`a2`. The topic-stance file is the group.
"""

from __future__ import annotations

import html
from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, pairwise, read_csv, spec

TASK = "persuasion/ukpconvarg"
DIRECTORY = "data/UKPConvArg1Strict-CSV"

SPEC = spec(
    key="ukpconvarg",
    dataset_id="UKPLab/acl2016-convincing-arguments (UKPConvArg1Strict)",
    revision="ca9d24e41b8805ff2e1d3585d552b49bb64a65f7",
    licence="cc-by-4.0",
    evidence="README.md at the pinned commit: data under CC-BY 4.0 (createdebate CC-BY "
    "3.0, convinceme public domain)",
    label_provenance="5 MTurk workers per pair; gold by MACE, Strict keeps agreeing pairs",
    tasks=(TASK,),
)

QUESTION = {
    "type": "choice",
    "instructions": (
        "The state gives a debate topic, a stance on it and two arguments written for "
        "that stance, Argument A and Argument B. Which argument is more convincing for "
        "this stance?"
    ),
    "criteria": {
        "A": "Argument A is more convincing for this stance.",
        "B": "Argument B is more convincing for this stance.",
    },
}


def readable(slug: str) -> str:
    return " ".join(slug.replace("-", " ").split())


def argument(text: str) -> str:
    return " ".join(html.unescape(text.replace("<br/>", " ")).split())


def candidates(root: Path) -> Iterator[HtCandidate]:
    for path in sorted((root / DIRECTORY).glob("*.csv")):
        relative = f"{DIRECTORY}/{path.name}"
        topic, _, stance = path.stem.partition("_")
        for index, row in enumerate(read_csv(path, "\t")):
            texts = [argument(row["a1"]), argument(row["a2"])]
            if row["label"] not in ("a1", "a2") or not all(texts):
                continue
            item_id = f"{path.stem}:{row['#id']}"
            shown, _, gold = pairwise(item_id, texts, int(row["label"][1]) - 1)
            state = {
                "topic": readable(topic),
                "stance": readable(stance),
                "argument_a": shown[0],
                "argument_b": shown[1],
            }
            item = make(
                SPEC,
                TASK,
                item_id,
                path.stem,
                "all",
                relative,
                index,
                state,
                dict(QUESTION),
                gold,
                overlap_texts=texts,
                option_texts=shown,
            )
            if item:
                yield item
