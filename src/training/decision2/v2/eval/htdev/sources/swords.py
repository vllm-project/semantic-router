"""Swords (backup for word meaning): does the substitute keep the sentence's meaning?

`swords-v1.1_{test,dev}.json.gz`: contexts, targets (word + character offset),
substitutes and their acceptability judgements. Gold: a strict majority of TRUE
(acceptable) = yes, of FALSE = no; ties are dropped. The target (one context and word)
is the group, so one substitute per target by default.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, noul, spec

TASK = "wordsense/swords"
FILES = (
    ("assets/parsed/swords-v1.1_test.json.gz", "test"),
    ("assets/parsed/swords-v1.1_dev.json.gz", "validation"),
)
QUESTION = noul(
    "The state gives a passage, a target word that occurs in it (at the given character "
    "offset of the passage), and a substitute word.",
    "Replacing the target word with the substitute keeps the meaning of the passage.",
    "Replacing the target word with the substitute changes the meaning of the passage.",
)

SPEC = spec(
    key="swords",
    dataset_id="p-lambda/swords (v1.1)",
    revision="04ca75370d0ce098a7f4db68240fc8e79a4f7b3b",
    licence="cc-by-3.0",
    evidence="README.md at the pinned commit: 'published under the permissive CC-BY-3.0-US "
    "license' (CoInCo and MASC content under the same licence)",
    label_provenance="crowd acceptability judgements per (context, target, substitute)",
    tasks=(TASK,),
)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for relative, split in FILES:
        with gzip.open(root / relative, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        contexts, targets = data["contexts"], data["targets"]
        for index, (sub_id, sub) in enumerate(data["substitutes"].items()):
            labels = data["substitute_labels"].get(sub_id) or []
            yes = sum(label == "TRUE" for label in labels)
            no = sum(label == "FALSE" for label in labels)
            target = targets.get(sub["target_id"])
            if yes == no or target is None:
                continue
            context = contexts[target["context_id"]]["context"]
            state = {
                "passage": context,
                "target_word": target["target"],
                "target_offset": target["offset"],
                "substitute": sub["substitute"],
            }
            item = make(
                SPEC,
                TASK,
                f"{split}:{sub_id}",
                sub["target_id"],
                split,
                relative,
                index,
                state,
                dict(QUESTION),
                yes > no,
                overlap_texts=[context],
            )
            if item:
                yield item
