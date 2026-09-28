"""Diplomacy (It Takes Two to Lie): does the sender intend this message to deceive?

Each dialogue row lists its messages with the sender's own truthful/lie annotation
(`sender_labels`, made at send time). One item per annotated message: up to three
preceding messages of the same dialogue plus the target message, with sender and
receiver (the players' countries). Receiver labels, game scores and seasons are never
shown. The game is the group (the prereg caps items per game).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, noul, read_jsonl, spec

TASK = "deception/diplomacy"
FILES = (
    ("data/test.jsonl", "test"),
    ("data/validation.jsonl", "validation"),
    ("data/train.jsonl", "train"),
)
CONTEXT = 3
QUESTION = noul(
    "The state is a message sent between two players of an online game of Diplomacy, "
    "with up to three messages that preceded it in their conversation.",
    "The sender intends this message to deceive the receiver.",
    "The sender does not intend this message to deceive the receiver.",
)

SPEC = spec(
    key="diplomacy",
    dataset_id="DenisPeskov/2020_acl_diplomacy",
    revision="c28c0870a639f6f388c87d4f8eaf685e64bfb1a6",
    licence="cc-by-4.0",
    evidence="LICENSE and README.md at the pinned commit: CC BY 4.0",
    label_provenance="senders marked each message truthful or deceptive when sending it",
    tasks=(TASK,),
)


def turn(row: dict, j: int) -> dict[str, str]:
    return {
        "from": str(row["speakers"][j]),
        "to": str(row["receivers"][j]),
        "text": str(row["messages"][j]).strip(),
    }


def candidates(root: Path) -> Iterator[HtCandidate]:
    for relative, split in FILES:
        for index, row in enumerate(read_jsonl(root / relative)):
            game = str(row["game_id"])
            for j, label in enumerate(row["sender_labels"]):
                if not isinstance(label, bool) or not str(row["messages"][j]).strip():
                    continue
                state = {
                    "previous_messages": [
                        turn(row, k)
                        for k in range(max(0, j - CONTEXT), j)
                        if str(row["messages"][k]).strip()
                    ],
                    "message": turn(row, j),
                }
                item = make(
                    SPEC,
                    TASK,
                    f"game{game}:msg{row['absolute_message_index'][j]}",
                    game,
                    split,
                    relative,
                    f"{index}:{j}",
                    state,
                    dict(QUESTION),
                    not label,
                    overlap_texts=[state["message"]["text"]],
                )
                if item:
                    yield item
