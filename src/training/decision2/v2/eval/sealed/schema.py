"""Candidate schema and validation for JevArena-C1 converters.

A converter module `v2/eval/sealed/sources/<key>.py` exports

    SPEC: SourceSpec
    def candidates(root: Path) -> Iterator[Candidate]

where `root` is the source's snapshot at `SPEC.revision` (an `hf download --revision`
local dir). It yields every eligible row once, already filtered (post-cutoff dates,
human-label configs only, model-generated columns dropped) and converted with one frozen
template per task. It never selects or samples: selection is central (`build`).

Label-blind conversion rules:

- One fixed template per task: instructions and criteria written from the dataset card
  or annotation guidelines, never tuned on labels or model outputs.
- Fixed label sets use semantic snake_case keys in one canonical order for every item.
  Per-item option sets (multiple choice, pairwise) are ordered by `display_order` (a hash
  of the item id, independent of the gold) and keyed A, B, C ... by display position.
- Noul criteria are exactly {"true": ..., "false": ...}; Score criteria are a list of
  level descriptions in ascending order and the gold is the level index.
- Model-generated labels, scores, rationales or "match" columns are never read.
- `state` holds only what the decision needs; it must not state or hint the label.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_INPUT_CHARS = 28_000
LONG_INPUT_CHARS = 4_000
TYPES = ("choice", "noul", "score")
LANGUAGE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
KEY = re.compile(r"^[a-z0-9][a-z0-9_]*$|^[A-Z]$")
CUTOFF = "2026-06-01"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    dataset_id: str
    revision: str
    licence: str
    licence_flag: str | None
    first_release: str
    evidence: str
    label_provenance: str
    languages: tuple[str, ...]
    tasks: tuple[str, ...]
    notes: str = ""


@dataclass
class Candidate:
    source: str
    task: str
    source_item_id: str
    group_id: str
    balance_label: str
    language: str
    state: Any
    question: dict[str, Any]
    gold: Any
    overlap_texts: list[str] = field(default_factory=list)
    date: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def display_order(item_id: str, count: int, salt: str = "c1-display") -> list[int]:
    """A gold-independent permutation of range(count) derived from the item id."""
    return sorted(
        range(count),
        key=lambda i: hashlib.sha256(f"{salt}|{item_id}|{i}".encode()).hexdigest(),
    )


def letters(count: int) -> list[str]:
    if not 2 <= count <= 26:
        raise ValueError("letter keys need 2..26 options")
    return [chr(ord("A") + i) for i in range(count)]


def input_chars(state: Any, question: dict[str, Any]) -> int:
    return len(
        json.dumps(
            {"state": state, "questions": {"decision": question}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate(candidate: Candidate, spec: SourceSpec) -> list[str]:
    """Problems with one candidate (empty list = valid)."""
    problems = []
    question = candidate.question
    qtype = question.get("type")
    if candidate.source != spec.key:
        problems.append("source key differs from SPEC.key")
    if candidate.task not in spec.tasks:
        problems.append(f"task {candidate.task!r} not in SPEC.tasks")
    if qtype not in TYPES:
        problems.append(f"bad type {qtype!r}")
    if (
        not isinstance(question.get("instructions"), str)
        or not question["instructions"].strip()
    ):
        problems.append("empty instructions")
    if set(question) - {"type", "instructions", "criteria"}:
        problems.append("extra question fields")
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
            problems.append("choice criteria must be a dict of 2..26 options")
        else:
            if not all(isinstance(k, str) and KEY.match(k) for k in criteria):
                problems.append("choice keys must be snake_case or single capitals")
            if not all(isinstance(v, str) and v.strip() for v in criteria.values()):
                problems.append("empty option description")
            if candidate.gold not in criteria:
                problems.append("gold not among choice keys")
    elif qtype == "noul":
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            problems.append("noul criteria must be exactly true/false")
        if not isinstance(candidate.gold, bool):
            problems.append("noul gold must be bool")
    elif qtype == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 11:
            problems.append("score criteria must be a list of 2..11 levels")
        elif (
            not isinstance(candidate.gold, int)
            or isinstance(candidate.gold, bool)
            or not 0 <= candidate.gold < len(criteria)
        ):
            problems.append("score gold must be a level index")
    if not LANGUAGE.match(candidate.language or ""):
        problems.append("language must be an ISO 639 code")
    if not candidate.source_item_id or not candidate.group_id:
        problems.append("missing source item or group id")
    if candidate.date is not None and candidate.date < CUTOFF:
        problems.append("row dated before the cutoff")
    try:
        size = input_chars(candidate.state, question)
    except (TypeError, ValueError):
        problems.append("state/question not JSON-serializable")
    else:
        if size > MAX_INPUT_CHARS:
            problems.append("input exceeds MAX_INPUT_CHARS")
    if isinstance(candidate.state, str) and not candidate.state.strip():
        problems.append("empty state")
    return problems
