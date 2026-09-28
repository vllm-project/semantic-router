"""JevArena-C1 converter for SpaceHunterInf/DeliChess.

Multi-party chess-puzzle deliberation chats. Every utterance has human labels
(trained Prolific annotators following the released guidelines) for
communicative function (9 fixed labels) and epistemic stance (3 fixed labels).
We emit two Choice tasks per utterance with the same state: the target utterance
plus a fixed window of preceding utterances from the same dialogue.

Only the ``human_*`` label columns are read. The ``gemini_*`` model predictions,
the ``*_match`` agreement flags (which would leak the human label) and the free
``annotator_note`` are never read: the CSV is parsed with an explicit column
allow-list.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from v2.eval.sealed.schema import MAX_INPUT_CHARS, Candidate, SourceSpec, input_chars

# Fixed context window: up to CONTEXT_TURNS preceding utterances of the same dialogue.
CONTEXT_TURNS = 8
COLUMNS = (
    "dialogue_id",
    "utterance_id",
    "utterance_position",
    "speaker_id",
    "text",
    "human_communicative_function",
    "human_epistemic_stance",
)

SPEC = SourceSpec(
    key="delichess",
    dataset_id="SpaceHunterInf/DeliChess",
    revision="277393d50567633ef1797eca0e76053bee10ca0d",
    licence="mit",
    licence_flag=None,
    first_release="2026-08-01",
    evidence=(
        "https://huggingface.co/api/datasets/SpaceHunterInf/DeliChess/commits/main"
        " — first commit (initial commit) 2026-08-01, data published 2026-08-01;"
        " HF createdAt 2026-08-01; card cites arXiv 2606.04987 (June 2026)."
    ),
    label_provenance=(
        "Trained proficient English speakers recruited via Prolific, following the"
        " released annotation guidelines (human_* columns; mostly one annotation per"
        " utterance). gemini_* predictions and *_match flags are model-derived and"
        " excluded."
    ),
    languages=("en",),
    tasks=("delichess/communicative_function", "delichess/epistemic_stance"),
    notes=(
        "Two Choice tasks share one state per utterance: the target utterance and"
        f" up to {CONTEXT_TURNS} preceding utterances of the same dialogue (speaker"
        " = dialogue-local anonymous id). group_id = dialogue (107 dialogues)."
        " Epistemic key 'unhedged' is the release's NO_HEDGED label. Utterances"
        " are short and context-dependent; no row dates (relative times only)."
    ),
)

FUNCTIONS = {
    "propose_answer": "Suggests a candidate move or answer.",
    "provide_reasoning": "Explains why a move or option may work or fail.",
    "explore_alternatives": (
        "Introduces, compares, or keeps open multiple candidate options."
    ),
    "request_reasoning": (
        "Asks another participant to explain, justify, or clarify their view."
    ),
    "evaluate_critique": "Assesses, supports, rejects, or compares candidate answers.",
    "agreement_alignment": (
        "Signals agreement, acceptance, or alignment with another participant."
    ),
    "coordinate_decision": (
        "Manages the group decision process, submission, or movement between"
        " puzzles."
    ),
    "social_moderation": (
        "Manages participation, turn-taking, inclusion, or dialogue dynamics."
    ),
    "social_offtask": (
        "Greeting, closing, joke, thanks, or unrelated social interaction."
    ),
}
STANCES = {
    "no_task_stance": (
        "No task-relevant stance is expressed, even after considering the local"
        " context (genuinely social or off-task content)."
    ),
    "hedged": (
        "A task-relevant stance marked as uncertain, tentative, hedged,"
        " question-framed, or weakly committed."
    ),
    "unhedged": (
        "A task-relevant stance without marked uncertainty, including assertions,"
        " answer proposals, contextual agreement, and unhedged task coordination."
    ),
}
_CONTEXT = (
    "The state is part of a text chat in which a small group deliberates over"
    " chess puzzles (each puzzle offers numbered candidate moves) before"
    " submitting answers. `current_utterance` is the utterance to label;"
    " `preceding_utterances` are the turns just before it, oldest first. Label"
    " the current utterance in its local dialogue context; do not judge whether"
    " any chess claim is correct."
)
FUNCTION_Q = {
    "type": "choice",
    "instructions": _CONTEXT
    + " What is the primary communicative function of the current utterance?",
    "criteria": FUNCTIONS,
}
STANCE_Q = {
    "type": "choice",
    "instructions": _CONTEXT
    + " Which epistemic stance does the current utterance express?",
    "criteria": STANCES,
}
_FUNCTION_GOLD = {key.upper(): key for key in FUNCTIONS}
_STANCE_GOLD = {
    "NO_TASK_STANCE": "no_task_stance",
    "HEDGED": "hedged",
    "NO_HEDGED": "unhedged",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = {name: header.index(name) for name in COLUMNS}
        return [{name: row[index[name]] for name in COLUMNS} for row in reader if row]


def from_rows(rows: Iterable[dict[str, str]]) -> Iterator[Candidate]:
    dialogues: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dialogues[row["dialogue_id"]].append(row)
    for dialogue_id in sorted(dialogues):
        turns = sorted(
            dialogues[dialogue_id], key=lambda r: int(r["utterance_position"])
        )
        for position, row in enumerate(turns):
            if not row["text"].strip():
                continue
            context = turns[max(0, position - CONTEXT_TURNS) : position]
            state = {
                "preceding_utterances": [
                    {"speaker": turn["speaker_id"], "text": turn["text"]}
                    for turn in context
                ],
                "current_utterance": {
                    "speaker": row["speaker_id"],
                    "text": row["text"],
                },
            }
            overlap = [row["text"]] + [turn["text"] for turn in context]
            for task, question, gold in (
                (
                    "delichess/communicative_function",
                    FUNCTION_Q,
                    _FUNCTION_GOLD.get(row["human_communicative_function"]),
                ),
                (
                    "delichess/epistemic_stance",
                    STANCE_Q,
                    _STANCE_GOLD.get(row["human_epistemic_stance"]),
                ),
            ):
                if gold is None or input_chars(state, question) > MAX_INPUT_CHARS:
                    continue
                yield Candidate(
                    source="delichess",
                    task=task,
                    source_item_id=row["utterance_id"],
                    group_id=dialogue_id,
                    balance_label=gold,
                    language="en",
                    state=state,
                    question=question,
                    gold=gold,
                    overlap_texts=list(overlap),
                    date=None,
                )


def candidates(root: Path) -> Iterator[Candidate]:
    yield from from_rows(read_rows(root / "data" / "utterances.csv"))
