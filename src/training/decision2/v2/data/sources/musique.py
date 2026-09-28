"""MuSiQue-Full v1.0 TRAIN as A3 rows over the 20-paragraph state: whether the
paragraphs answer the question (Noul; the answerable and unanswerable twins of
an id form one group) and which paragraph supports the last decomposition step
(Choice, answerable twin only).

Choice options name paragraphs in state order and are not rotated, so their
gold positions are reported by the builder, never rebalanced.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

from v2.data.sources.common import cap_groups, choice_options, make_row, noul_options

ARM = "a3"
SOURCE = "musique_full_v1.0_train"
NOUL_FAMILY = "musique_answerable"
CHOICE_FAMILY = "musique_final_support"
SEED = "a3-musique-v1"
PAIRS = 1000
CAPS = {"musique_pairs": PAIRS}
DATA = "musique_full_v1.0_train.jsonl"
NOUL_INSTRUCTIONS = (
    "Do these paragraphs contain enough information to answer this question? "
    "Question: {}"
)
CHOICE_INSTRUCTIONS = (
    "Which paragraph states the fact needed for the final step of answering "
    "this question? Question: {}"
)


def _clean(text: str) -> str:
    return " ".join(text.split())


def paragraph_names(paragraphs: list[dict[str, Any]]) -> list[str]:
    return [
        f"Paragraph {position} — {_clean(paragraph['title'])}"
        for position, paragraph in enumerate(paragraphs, 1)
    ]


def render(paragraphs: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{name}: {_clean(paragraph['paragraph_text'])}"
        for name, paragraph in zip(paragraph_names(paragraphs), paragraphs)
    )


def final_support(item: dict[str, Any]) -> int:
    target = item["question_decomposition"][-1]["paragraph_support_idx"]
    hits = [
        position
        for position, paragraph in enumerate(item["paragraphs"])
        if target is not None and paragraph["idx"] == target
    ]
    if len(hits) != 1:
        raise ValueError(
            f"MuSiQue {item['id']}: final support idx {target!r} matches {len(hits)} paragraphs"
        )
    return hits[0]


def twins(items: list[dict[str, Any]]) -> dict[str, dict[bool, dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        if type(item["answerable"]) is not bool:
            raise ValueError(f"MuSiQue {item['id']}: answerable must be a boolean")
        by_id[item["id"]].append(item)
    broken = sorted(
        key
        for key, members in by_id.items()
        if sorted(member["answerable"] for member in members) != [False, True]
    )
    if broken:
        raise ValueError(
            f"{len(broken)} MuSiQue ids lack exactly one answerable and one "
            f"unanswerable twin, e.g. {broken[:3]}"
        )
    return {
        key: {member["answerable"]: member for member in members}
        for key, members in by_id.items()
    }


def build(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    with (root / DATA).open(encoding="utf-8") as stream:
        pairs = twins([json.loads(line) for line in stream if line.strip()])
    rows: dict[str, list[dict[str, Any]]] = {NOUL_FAMILY: [], CHOICE_FAMILY: []}
    for key in sorted(pairs):
        for answerable in (True, False):
            item = pairs[key][answerable]
            question = _clean(item["question"])
            shared = {
                "arm": ARM,
                "source": SOURCE,
                "language": "en",
                "group_key": key,
                "local_id": f"{key}:{'answerable' if answerable else 'unanswerable'}",
                "state": render(item["paragraphs"]),
            }
            rows[NOUL_FAMILY].append(
                make_row(
                    family=NOUL_FAMILY,
                    task_type="noul",
                    instructions=NOUL_INSTRUCTIONS.format(question),
                    options=noul_options("en"),
                    label=int(answerable),
                    render_template=f"{NOUL_FAMILY}/v1",
                    audit={"musique_id": key, "answerable": answerable},
                    **shared,
                )
            )
            if answerable:
                gold = final_support(item)
                rows[CHOICE_FAMILY].append(
                    make_row(
                        family=CHOICE_FAMILY,
                        task_type="choice",
                        instructions=CHOICE_INSTRUCTIONS.format(question),
                        options=choice_options(paragraph_names(item["paragraphs"])),
                        label=gold,
                        render_template=f"{CHOICE_FAMILY}/v1",
                        audit={
                            "musique_id": key,
                            "support_idx": item["paragraphs"][gold]["idx"],
                            "hops": len(item["question_decomposition"]),
                        },
                        **shared,
                    )
                )
    return rows, {
        "inputs": {DATA: file_sha256(root / DATA)},
        "twin_pairs": len(pairs),
        "dropped": {NOUL_FAMILY: {}, CHOICE_FAMILY: {}},
    }


def select(
    rows: dict[str, list[dict[str, Any]]], pairs: int = PAIRS, seed: str = SEED
) -> dict[str, list[dict[str, Any]]]:
    """Whole twin groups: `pairs` groups of two Noul rows plus their Choice row."""
    noul = cap_groups(rows[NOUL_FAMILY], 2 * pairs, seed)
    kept = {row["group_id"] for row in noul}
    return {
        NOUL_FAMILY: noul,
        CHOICE_FAMILY: [row for row in rows[CHOICE_FAMILY] if row["group_id"] in kept],
    }
