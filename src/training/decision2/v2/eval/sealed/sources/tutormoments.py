"""JevArena-C1 converter for allenai/tutormoments-preview.

Real K-12 tutoring sessions (mostly maths) with human turn-level annotations. In each
annotation pass a human annotator marks key moments of one type over a session:
``scaffolding`` (the tutor scaffolds, or misses a chance to scaffold, the
student's problem solving) or ``rapport`` (rapport building or rupture). We emit
one Choice per distinct human-marked moment: which type of key moment is it?

Only the human ``annotations`` config (moment type and turn range) and the
``transcripts`` config (turn text) are read. The annotators' situation / action /
result descriptions, captions and session summaries are never read (they describe
the moment and would leak its type), nor are the Gemini ``enrichments``, the
``moments`` benchmark set, the LLM-assisted ``ground_truth`` or the synthetic
``benchmark_520`` files (the last two are not even downloaded).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import MAX_INPUT_CHARS, Candidate, SourceSpec, input_chars

# Fixed context window: the marked moment (turn_number_start..turn_number_end) in
# full, plus the preceding session turns up to CONTEXT_CHARS characters of turn text
# (whole turns, nearest first). Moments that do not fit MAX_INPUT_CHARS are dropped.
CONTEXT_CHARS = 2_000
TYPES = ("scaffolding", "rapport")
# Session language (the release leaves primary_language empty): Spanish if at least
# half of the listed function words in the session are Spanish, else English.
_EN = frozenset(
    "the and you is to it that what of this do can okay yeah so are how".split()
)
_ES = frozenset(
    "el la que de es y en lo los las por para una un qué cómo sí muy bien pero"
    " está tienes puedes".split()
)
_WORD = re.compile(r"[a-záéíóúñü]+")

SPEC = SourceSpec(
    key="tutormoments",
    dataset_id="allenai/tutormoments-preview",
    revision="66058b4c7ef5e7631b3c4d39a1dfa8172e36d8bc",
    licence="cc-by-4.0",
    licence_flag=None,
    first_release="2026-07-06",
    evidence=(
        "https://huggingface.co/api/datasets/allenai/tutormoments-preview/commits/main"
        " — first commit (initial commit) 2026-07-06, data uploaded 2026-07-06; HF"
        " createdAt 2026-07-06; paper in preparation (no arXiv id on the card)."
    ),
    label_provenance=(
        "Human annotator passes over real sessions (annotations config): each pass"
        " marks scaffolding or rapport key moments as turn ranges; v2_preselected"
        " passes re-annotate moments placed by another human annotator. ground_truth"
        " (LLM-assisted), benchmark_520 (synthetic) and moments (benchmark set) are"
        " excluded."
    ),
    languages=("en", "es"),
    tasks=("tutormoments/moment_type", "tutormoments/is_rapport"),
    notes=(
        "Each distinct moment goes to exactly one task by the parity of the SHA-256"
        " of its item id (label-independent): even -> Choice moment_type, odd -> Noul"
        " is_rapport (the same two human classes as a yes/no question)."
        " One item per distinct (transcript, type, start, end) moment; duplicate"
        " marks from several passes collapse into one item, and moments whose turn"
        " range overlaps a moment of the other type in the same session are"
        " excluded as ambiguous. state = {preceding_turns (up to"
        f" {CONTEXT_CHARS} chars of whole turns), moment (all marked turns)}};"
        " oversize moments are dropped, never cut. group_id = transcript. Transcripts"
        " were transcribed by Gemini 2.5 Pro (ASR text, real sessions). Annotation"
        " timestamps (2025-11..2026-06) are internal work dates of unpublished"
        " private sessions, not publication dates, so date=None. Only 23 of 207"
        " sessions received both pass types. A presence Noul was not built: passes"
        " mark a median ~11-28% of session turns, so unmarked turns are not"
        " confirmed negatives."
    ),
)

_SESSION = (
    "The state is an excerpt from a real one-to-one K-12 tutoring session, mostly"
    " mathematics (automatically transcribed speech; speaker labels may contain"
    " transcription errors). `moment` is a span of turns that a human annotator"
    " marked as a key moment of the session; `preceding_turns` are the turns just"
    " before it. The annotator marked it for one of two kinds of key moment; judge"
    " which kind the span is primarily about."
)
MOMENT_Q = {
    "type": "choice",
    "instructions": _SESSION + " Which kind of key moment is the marked span?",
    "criteria": {
        "scaffolding": (
            "A scaffolding moment: the tutor supports, or misses a chance to support,"
            " the student's own problem solving (hints, prompts, questions,"
            " breaking the problem into steps, or giving away the answer)."
        ),
        "rapport": (
            "A rapport moment: the tutor builds, maintains, or ruptures the"
            " relationship with the student, or misses a chance to respond to it"
            " (encouragement, empathy, personal or social talk, reactions to the"
            " student's feelings or engagement)."
        ),
    },
}
RAPPORT_Q = {
    "type": "noul",
    "instructions": _SESSION + " Is it a rapport moment?",
    "criteria": {
        "true": MOMENT_Q["criteria"]["rapport"],
        "false": MOMENT_Q["criteria"]["scaffolding"],
    },
}


def task_of(item_id: str) -> str:
    """Label-independent split of moments between the Choice and the Noul task."""
    parity = int(hashlib.sha256(item_id.encode("utf-8")).hexdigest(), 16) % 2
    return "tutormoments/is_rapport" if parity else "tutormoments/moment_type"


def session_language(turns: list[dict]) -> str:
    english = spanish = 0
    for turn in turns:
        words = _WORD.findall((turn.get("text") or "").lower())
        english += sum(word in _EN for word in words)
        spanish += sum(word in _ES for word in words)
    return "es" if spanish and spanish >= english else "en"


def _render(turn: dict) -> dict:
    return {"turn": turn["turn_number"], "speaker": turn["role"], "text": turn["text"]}


def candidates(root: Path) -> Iterator[Candidate]:
    sessions: dict[str, list[dict]] = {}
    with (root / "transcripts.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                sessions[record["transcript_id"]] = sorted(
                    record["turns"], key=lambda turn: turn["turn_number"]
                )
    marked: dict[tuple[str, str, int, int], None] = {}
    spans: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with (root / "annotations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("annotation_type")
            transcript_id = record.get("transcript_id")
            if kind not in TYPES or transcript_id not in sessions:
                continue
            count = len(sessions[transcript_id])
            for mark in record.get("turn_annotations") or []:
                start = mark.get("turn_number_start")
                end = mark.get("turn_number_end")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                if not 1 <= start <= end <= count:
                    continue
                if (transcript_id, kind, start, end) not in marked:
                    marked[(transcript_id, kind, start, end)] = None
                    spans[transcript_id][kind].append((start, end))
    languages = {tid: session_language(turns) for tid, turns in sessions.items()}
    for transcript_id, kind, start, end in sorted(marked):
        other = TYPES[1 - TYPES.index(kind)]
        if any(s <= end and start <= e for s, e in spans[transcript_id][other]):
            continue
        turns = sessions[transcript_id]
        index = {turn["turn_number"]: i for i, turn in enumerate(turns)}
        first, last = index[start], index[end]
        before: list[dict] = []
        used = 0
        for turn in reversed(turns[:first]):
            used += len(turn["text"])
            if used > CONTEXT_CHARS:
                break
            before.insert(0, turn)
        moment = turns[first : last + 1]
        state = {
            "preceding_turns": [_render(turn) for turn in before],
            "moment": [_render(turn) for turn in moment],
        }
        item_id = f"{transcript_id}:{start}-{end}"
        task = task_of(item_id)
        question, gold = (
            (RAPPORT_Q, kind == "rapport")
            if task == "tutormoments/is_rapport"
            else (MOMENT_Q, kind)
        )
        if input_chars(state, question) > MAX_INPUT_CHARS:
            continue
        yield Candidate(
            source="tutormoments",
            task=task,
            source_item_id=item_id,
            group_id=transcript_id,
            balance_label=str(gold),
            language=languages[transcript_id],
            state=state,
            question=question,
            gold=gold,
            overlap_texts=[turn["text"] for turn in before + moment if turn["text"]],
            date=None,
        )
