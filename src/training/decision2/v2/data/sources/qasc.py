"""QASC TRAIN as A1 Choice rows: the two annotated facts are the state (never
the combined fact), the question is the instruction, and the eight answer
choices are the rotated options. Items that share a normalized fact pair form
one group.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

from v2.data.sources.common import choice_options, make_row, rotate

ARM = "a1"
SOURCE = "qasc_train"
FAMILY = "qasc_facts"
SEED = "a1-qasc-v1"
CAPS = {FAMILY: 2000}
DATA = "data/train.jsonl"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _clean(text: str) -> str:
    return " ".join(text.split())


def _choice_row(descriptions: list[str], gold: int, **fields: Any) -> dict[str, Any]:
    probe = make_row(options=choice_options(descriptions), label=gold, **fields)
    options, label = rotate(probe["options"], gold, f"{fields['arm']}-v1:{probe['id']}")
    return make_row(options=options, label=label, **fields)


def build(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    with (root / DATA).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in seen:
                raise ValueError(f"duplicate QASC id {item['id']}")
            seen.add(item["id"])
            texts = [_clean(text) for text in item["choices"]["text"]]
            keys = item["choices"]["label"]
            if (
                len(texts) != len(keys)
                or len(texts) < 2
                or not all(texts)
                or item["answerKey"] not in keys
            ):
                dropped["malformed_choices"] += 1
                continue
            gold = keys.index(item["answerKey"])
            normalized = [normalize(text) for text in texts]
            if normalized.count(normalized[gold]) > 1:
                dropped["gold_text_duplicated"] += 1
                continue
            fact1, fact2, question = (
                _clean(item[field]) for field in ("fact1", "fact2", "question")
            )
            if not (fact1 and fact2 and question):
                dropped["empty_fact_or_question"] += 1
                continue
            rows.append(
                _choice_row(
                    texts,
                    gold,
                    arm=ARM,
                    source=SOURCE,
                    family=FAMILY,
                    task_type="choice",
                    language="en",
                    group_key=f"{normalize(fact1)}|{normalize(fact2)}",
                    local_id=item["id"],
                    state=f"Fact 1: {fact1}\nFact 2: {fact2}",
                    instructions=question,
                    render_template=f"{FAMILY}/v1",
                    audit={"qasc_id": item["id"], "answer_key": item["answerKey"]},
                )
            )
    return {FAMILY: rows}, {
        "inputs": {DATA: file_sha256(root / DATA)},
        "dropped": {FAMILY: dict(sorted(dropped.items()))},
    }
