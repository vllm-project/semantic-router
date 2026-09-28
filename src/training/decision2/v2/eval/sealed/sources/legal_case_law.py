"""JevArena-C1 converter for lbrenap1/mining-legal-arguments-us-corporate-case-law.

42 U.S. federal tax opinions (I.R.C. s.368 corporate reorganisations, 1935-1987)
with span-level functional labels by two law students, adjudicated by a law
professor. The release projects the adjudicated spans onto sentences. We emit one
Choice per labelled sentence: the argumentative function of the marked target
sentence, shown with a fixed window of the surrounding opinion text.

Read: ``sentences`` (text, order, primary label) and ``sentence_node_links``
(overlap geometry). The ``Unlabeled`` residue (no overlapping span) is not a
functional judgement and is skipped; the retrieval and iaa configs are not read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from v2.eval.sealed.schema import MAX_INPUT_CHARS, Candidate, SourceSpec, input_chars

# Fixed context window: whole sentences before and after the target, nearest first,
# up to CONTEXT_CHARS characters of sentence text on each side.
CONTEXT_CHARS = 2_000
# A sentence is eligible only if every overlapping span carries the same label and
# its primary span covers at least MIN_COVERAGE of the sentence's characters.
MIN_COVERAGE = 0.5
LABELS = {
    "Rule": "rule",
    "Analysis": "analysis",
    "Conclusion": "conclusion",
    "Background Facts": "background_facts",
    "Procedural History": "procedural_history",
}

SPEC = SourceSpec(
    key="legal_case_law",
    dataset_id="lbrenap1/mining-legal-arguments-us-corporate-case-law",
    revision="05bf4bb8c3bf8f44957474b2965e2ad0d7700707",
    licence="cc-by-4.0",
    licence_flag=None,
    first_release="2026-08-30",
    evidence=(
        "https://huggingface.co/api/datasets/lbrenap1/"
        "mining-legal-arguments-us-corporate-case-law/commits/main — first commit"
        " (initial commit) 2026-08-30, v1.0.0 published 2026-08-30; HF createdAt"
        " 2026-08-30; registry cites arXiv 2609.25441 (the card says the paper link"
        " is pending)."
    ),
    label_provenance=(
        "Expert annotation: two law students labelled spans; a law professor"
        " adjudicated the final case representations (10 cases double-annotated,"
        " iaa_* configs). Sentence labels are a deterministic projection of the"
        " adjudicated human spans."
    ),
    languages=("en",),
    tasks=("legal_case_law/argument_function",),
    notes=(
        "Old text: public U.S. tax opinions from 1935-1987 (Westlaw copies) that"
        " are very likely in pretraining corpora; only the labels are new. group_id"
        " = opinion (42 cases), so source groups are few. state = {context_before,"
        f" target_sentence, context_after}} with up to {CONTEXT_CHARS} chars of whole"
        " sentences on each side. Unlabeled sentences are skipped; sentences with"
        " overlapping spans of different labels or primary-span coverage below"
        f" {MIN_COVERAGE:.0%} are excluded as ambiguous. Opinion years are not"
        " publication dates of the labels, so date=None."
    ),
)

ARGUMENT_Q = {
    "type": "choice",
    "instructions": (
        "The state is an excerpt from a U.S. federal court opinion in a corporate"
        " tax (reorganisation) case. `target_sentence` is the sentence to classify;"
        " `context_before` and `context_after` are the opinion text immediately"
        " around it. What function does the target sentence serve in the court's"
        " opinion?"
    ),
    "criteria": {
        "rule": (
            "A generally applicable statement: a legal rule, test, abstract"
            " criterion, or precedent cited as authority."
        ),
        "analysis": (
            "Case-specific reasoning that applies or interprets a rule using the"
            " present facts or record, including intermediate conclusions."
        ),
        "conclusion": (
            "The terminal outcome of the court's argument on an issue, not an"
            " intermediate step of the reasoning."
        ),
        "background_facts": (
            "Facts and transaction details that set the scene (corporate structure,"
            " ownership, transfers, business operations, administrative events of"
            " the dispute) without directly taking part in the reasoning."
        ),
        "procedural_history": (
            "Litigation posture and court procedure, such as deficiency notices,"
            " refund claims, appeals, remands, and the decision under review."
        ),
    },
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _nearest(texts: list[str]) -> list[str]:
    """Leading whole texts whose joined length stays within CONTEXT_CHARS."""
    kept: list[str] = []
    used = 0
    for text in texts:
        used += len(text) + 1
        if used > CONTEXT_CHARS:
            break
        kept.append(text)
    return kept


def candidates(root: Path) -> Iterator[Candidate]:
    sentences = _read_jsonl(root / "data" / "sentences.jsonl")
    links: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for link in _read_jsonl(root / "data" / "sentence_node_links.jsonl"):
        links[(link["case_id"], link["passage_id"])].append(link)
    cases: dict[str, list[dict]] = defaultdict(list)
    for sentence in sentences:
        cases[sentence["case_id"]].append(sentence)
    for case_id in sorted(cases):
        ordered = sorted(cases[case_id], key=lambda sentence: sentence["order"])
        texts = [sentence["text"] for sentence in ordered]
        for position, sentence in enumerate(ordered):
            gold = LABELS.get(sentence["label"])
            if gold is None:
                continue
            overlapping = links[(case_id, sentence["passage_id"])]
            if len({link["label"] for link in overlapping}) != 1:
                continue
            primary = [
                link
                for link in overlapping
                if link["node_id"] == sentence.get("source_node_id")
            ]
            span = max(1, sentence["end"] - sentence["start"])
            if not primary or primary[0]["overlap_characters"] < MIN_COVERAGE * span:
                continue
            state = {
                "context_before": " ".join(reversed(_nearest(texts[:position][::-1]))),
                "target_sentence": sentence["text"],
                "context_after": " ".join(_nearest(texts[position + 1 :])),
            }
            if input_chars(state, ARGUMENT_Q) > MAX_INPUT_CHARS:
                continue
            yield Candidate(
                source="legal_case_law",
                task="legal_case_law/argument_function",
                source_item_id=f"{case_id}:{sentence['passage_id']}",
                group_id=case_id,
                balance_label=gold,
                language="en",
                state=state,
                question=ARGUMENT_Q,
                gold=gold,
                overlap_texts=[
                    text
                    for text in (
                        state["context_before"],
                        sentence["text"],
                        state["context_after"],
                    )
                    if text
                ],
                date=None,
            )
