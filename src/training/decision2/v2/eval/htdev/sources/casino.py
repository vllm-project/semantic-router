"""CaSiNo (backup for deception): how satisfied was participant A with the outcome?

One item per dialogue about its first participant (`mturk_agent_1`, shown as A; the
other is B), so the choice of participant never depends on the gold. State: the chat,
with deal actions rendered as text. Gold: the participant's own post-negotiation
satisfaction (5 levels). Priorities, reasons, points, demographics and strategy
annotations are never shown. Single split (`all`); the dialogue is the group.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, make, read_parquet, score, spec

TASK = "outcome/casino"
FILE = "data/train-00000-of-00001.parquet"
LEVELS = [
    "Extremely dissatisfied",
    "Slightly dissatisfied",
    "Undecided",
    "Slightly satisfied",
    "Extremely satisfied",
]
NAMES = {"mturk_agent_1": "A", "mturk_agent_2": "B"}
QUESTION = score(
    "The state is a negotiation chat between two campers, A and B, who split packages "
    "of food, water and firewood. How satisfied was A with the outcome of this "
    "negotiation?",
    [level.lower() for level in LEVELS],
)

SPEC = spec(
    key="casino",
    dataset_id="kchawla123/casino",
    revision="290898d2d08b6591db17005504e40ce00ac1028e",
    licence="cc-by-4.0",
    evidence="README.md at the pinned revision: licence cc-by-4.0",
    label_provenance="participants' own post-negotiation satisfaction self-report",
    tasks=(TASK,),
)


def message(turn: dict) -> str:
    text = str(turn.get("text") or "").strip()
    if text == "Submit-Deal":
        data = turn.get("task_data") or {}
        mine = data.get("issue2youget") or {}
        theirs = data.get("issue2theyget") or {}
        parts = ", ".join(f"{k} {mine.get(k, '')}".strip() for k in sorted(mine))
        other = ", ".join(f"{k} {theirs.get(k, '')}".strip() for k in sorted(theirs))
        return f"[proposes a deal: I get {parts}; you get {other}]"
    if text == "Accept-Deal":
        return "[accepts the deal]"
    if text == "Reject-Deal":
        return "[rejects the deal]"
    if text == "Walk-Away":
        return "[walks away without a deal]"
    return text


def candidates(root: Path) -> Iterator[HtCandidate]:
    rows = read_parquet(root / FILE, ["chat_logs", "participant_info"])
    for index, row in enumerate(rows):
        info = (row["participant_info"] or {}).get("mturk_agent_1") or {}
        satisfaction = (info.get("outcomes") or {}).get("satisfaction")
        if satisfaction not in LEVELS:
            continue
        chat = [
            {"speaker": NAMES.get(turn.get("id"), "?"), "text": message(turn)}
            for turn in row["chat_logs"] or []
            if message(turn)
        ]
        if not chat:
            continue
        item = make(
            SPEC,
            TASK,
            f"dialogue{index}",
            f"dialogue{index}",
            "all",
            FILE,
            index,
            {"chat": chat},
            dict(QUESTION),
            LEVELS.index(satisfaction),
            overlap_texts=[turn["text"] for turn in chat],
        )
        if item:
            yield item
