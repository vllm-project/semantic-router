"""HalluTruthQA-4K test split: hallucination detection (Noul) and find-the-truth (Choice).

Each test question feeds exactly one of the two tasks (sha256 parity of its id).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from v2.eval.sealed.schema import (
    MAX_INPUT_CHARS,
    Candidate,
    SourceSpec,
    display_order,
    input_chars,
    letters,
    normalized,
)

SPEC = SourceSpec(
    key="hallutruthqa",
    dataset_id="Bekhouche/HalluTruthQA-4K",
    revision="e6e2b14733effc8da7b87fa79230a20cc4b7336b",
    licence="cc-by-nc-nd-4.0",
    licence_flag="NC-ND",
    first_release="2026-08-01",
    evidence=(
        "https://huggingface.co/api/datasets/Bekhouche/HalluTruthQA-4K/commits/main: "
        "repo created 2026-07-27 with train/dev only, test split added 2026-08-10 "
        "(ae514f1f). Codabench 16390 (HalluScoring 2026 subtask 2.2): development "
        "phase from 2026-05-19 (train/dev), evaluation phase 2026-08-01..08-12 (test "
        "inputs). arXiv 2607.20219 v1 2026-07-22 (base), 2608.03966 v1 2026-08-04 (4K)."
    ),
    label_provenance=(
        "Four domain experts wrote and verified the questions and reference answers, "
        "labelled every Fanar-1-9B-Instruct answer and built the six candidate answers "
        "and answer key by hand; two trained assistants re-verified every record (kappa "
        "0.92 on the label, 0.976 exact agreement on the key before adjudication); the "
        "experts adjudicated disagreements (arXiv 2608.03966 sec. 3.1)."
    ),
    languages=("ar",),
    tasks=("hallutruthqa/hallucination", "hallutruthqa/find_truth"),
    notes=(
        "Test split only; test questions that also occur in the public train/dev splits "
        "are dropped. Disjoint questions per task: even sha256(id) -> hallucination "
        "(state: question + generated answer), odd -> find_truth (state: question only, "
        "because a correct generated answer names the right option); duplicate test "
        "questions follow their smallest id and share a group. Options are re-ordered "
        "with display_order (source answer letters are skewed). Risks: one generator "
        "model; public train/dev; hallucinated answers are about twice as long; the "
        "correct option is often the longest of the six."
    ),
)

NOUL = "hallutruthqa/hallucination"
CHOICE = "hallutruthqa/find_truth"
GOLD = {"hallucination": True, "no_hallucination": False}
NOUL_INSTRUCTIONS = (
    "The state holds a question in Arabic and an answer to it written by an AI "
    "assistant. Does the answer contain a factual hallucination, that is, at least one "
    "claim that is factually incorrect, fabricated, unsupported or misleading (for "
    "example a wrong name, date, number, quotation, citation or source)? An answer "
    "whose main point is correct still counts as hallucinated if any other claim in it "
    "is wrong; differences in wording, transliteration, level of detail or date format "
    "are not errors."
)
NOUL_CRITERIA = {
    "true": (
        "The answer contains at least one factually incorrect, fabricated, "
        "unsupported or misleading claim."
    ),
    "false": "Every factual claim in the answer is correct.",
}
CHOICE_INSTRUCTIONS = (
    "The state holds a question in Arabic. Which option answers the question "
    "correctly? Exactly one option is correct."
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _noul(row: dict[str, Any], group: str) -> Candidate:
    state = {
        "question": row["question"].strip(),
        "answer": row["generated_answer"].strip(),
    }
    gold = GOLD[row["label"]]
    return Candidate(
        source=SPEC.key,
        task=NOUL,
        source_item_id=row["id"],
        group_id=group,
        balance_label=str(gold),
        language="ar",
        state=state,
        question={
            "type": "noul",
            "instructions": NOUL_INSTRUCTIONS,
            "criteria": dict(NOUL_CRITERIA),
        },
        gold=gold,
        overlap_texts=[state["question"], state["answer"]],
    )


def _choice(row: dict[str, Any], group: str) -> Candidate:
    source = sorted(row["options"])
    order = display_order(row["id"], len(source))
    keys = letters(len(source))
    criteria = {
        keys[position]: str(row["options"][source[index]]).strip()
        for position, index in enumerate(order)
    }
    gold = keys[order.index(source.index(row["answer"]))]
    state = {"question": row["question"].strip()}
    return Candidate(
        source=SPEC.key,
        task=CHOICE,
        source_item_id=row["id"],
        group_id=group,
        balance_label=gold,
        language="ar",
        state=state,
        question={
            "type": "choice",
            "instructions": CHOICE_INSTRUCTIONS,
            "criteria": criteria,
        },
        gold=gold,
        overlap_texts=[state["question"]],
    )


def candidates(root: Path) -> Iterator[Candidate]:
    public = {
        normalized(row["question"])
        for split in ("train", "dev")
        for row in _rows(root / f"{split}.jsonl")
    }
    test = [
        row
        for row in _rows(root / "test.jsonl")
        if normalized(row["question"]) not in public
    ]
    first_id: dict[str, str] = {}
    for row in test:
        key = normalized(row["question"])
        first_id[key] = min(first_id.get(key, row["id"]), row["id"])
    for row in test:
        key = normalized(row["question"])
        build = _noul if int(_hash(first_id[key]), 16) % 2 == 0 else _choice
        candidate = build(row, "q-" + _hash(key)[:16])
        if input_chars(candidate.state, candidate.question) <= MAX_INPUT_CHARS:
            yield candidate
