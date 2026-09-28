"""JevArena-C1 converter for CLS-Lab/narrative-gold-annotations.

Human narrative annotations of short passages sampled from the Dolma corpus. Only
the ``*_gold`` columns (the primary annotator's final label) are read; the
secondary ``*_annotator_*`` columns are never loaded. We emit two Score tasks on
the setting config (concreteness, temporal grounding; 1-5 Likert, gold = level
index) and one Choice task on the event-relation config (causal relation between
two marked events).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from v2.eval.sealed.schema import (
    MAX_INPUT_CHARS,
    Candidate,
    SourceSpec,
    input_chars,
    normalized,
)

SETTING_COLUMNS = (
    "safe_instance_id",
    "sampled_text",
    "setting_concreteness_gold",
    "setting_temporal_grounding_gold",
)
EVENT_COLUMNS = (
    "safe_instance_id",
    "sampled_text",
    "assigned_span1",
    "assigned_span2",
    "span1_is_event_gold",
    "span2_is_event_gold",
    "causality_rating_gold",
)

SPEC = SourceSpec(
    key="narrative_gold",
    dataset_id="CLS-Lab/narrative-gold-annotations",
    revision="f22dd4d5cc18c10cc38984fe0600b21815190e92",
    licence="cc-by-4.0",
    licence_flag=None,
    first_release="2026-06-17",
    evidence=(
        "https://huggingface.co/api/datasets/CLS-Lab/narrative-gold-annotations/"
        "commits/main — first commit (initial commit + upload) 2026-06-17; HF"
        " createdAt 2026-06-17; card revisions of 2026-06-19 cite arXiv 2606.19468."
    ),
    label_provenance=(
        "Human annotation; per the 2026-06-19 card one author annotated all"
        " passages (the released *_gold labels), with second annotators on subsets"
        " for agreement (setting mean alpha 0.70, events kappa 0.68). Only *_gold"
        " columns are used."
    ),
    languages=("en",),
    tasks=(
        "narrative_gold/setting_concreteness",
        "narrative_gold/setting_temporal_grounding",
        "narrative_gold/event_causality",
        "narrative_gold/span_is_event",
    ),
    notes=(
        "Old text: passages are sampled from Dolma (Common Crawl, C4, Reddit,"
        " Gutenberg, Wikipedia), which is in many pretraining sets; only the labels"
        " are new. Sibling public sets (CLS-Lab/narrative-llm-annotations,"
        " CLS-Lab/narradolma) carry model labels on the same dimensions over Dolma"
        " passages. group_id = passage (safe_instance_id), shared by the setting"
        " and event tasks of one passage. Score levels map Likert 1..5 to index"
        " 0..4; the card publishes no level anchors, so the level wording follows the"
        " dimension names. Setting passages whose normalised text occurs more than"
        " once in the config are dropped (every copy). Event rows need a gold causal"
        " rating (present only when both spans are gold events); the two event spans"
        " are marked inline at their offsets. span_is_event (Noul): each assigned"
        " span of an event-relation row with a gold event judgement, marked alone"
        " inline; both spans of a passage share its group. date=None (Dolma crawl"
        " dates are not label dates)."
    ),
)

_PASSAGE = "The state holds a short English passage (`passage`) from a web text corpus."
CONCRETENESS_Q = {
    "type": "score",
    "instructions": _PASSAGE
    + " How concrete is the setting of the passage, i.e. how far does it depict a"
    " specific, tangible situation (particular places, objects, people or physical"
    " details) rather than abstract or general statements?",
    "criteria": [
        "Not concrete: abstract or general content with no specific setting.",
        "Slightly concrete: a vague or minimal setting.",
        "Moderately concrete: some specific details of the setting.",
        "Quite concrete: a clearly specified setting with several tangible details.",
        "Highly concrete: a vivid, specific setting rich in tangible detail.",
    ],
}
TEMPORAL_Q = {
    "type": "score",
    "instructions": _PASSAGE
    + " How strongly is the passage grounded in time, i.e. how far does it situate"
    " what it describes at specific times or in a clear temporal sequence (dates,"
    " times, durations, before/after relations)?",
    "criteria": [
        "Not grounded in time: no temporal information.",
        "Slightly grounded: vague or implicit temporal cues only.",
        "Moderately grounded: some explicit temporal references.",
        "Well grounded: clear temporal anchoring or sequencing.",
        "Strongly grounded: precise, pervasive temporal anchoring such as specific"
        " dates, times or an explicit timeline.",
    ],
}
CAUSALITY_Q = {
    "type": "choice",
    "instructions": (
        _PASSAGE + " Two event mentions are marked in it as <event1>...</event1> and"
        " <event2>...</event2> (their text is repeated in `event_1` and `event_2`)."
        " What is the causal relation between the two marked events?"
    ),
    "criteria": {
        "direct_cause": "One event directly causes the other.",
        "enables": (
            "One event enables the other (makes it possible or sets up the"
            " conditions for it) without directly causing it."
        ),
        "not_related": "Neither event causes nor enables the other.",
    },
}
EVENT_Q = {
    "type": "noul",
    "instructions": (
        _PASSAGE + " One span of it is marked as <span>...</span> (its text is repeated"
        " in `span`). Does the marked span describe an event, i.e. something that"
        " happens, is done or changes in the passage?"
    ),
    "criteria": {
        "true": "The marked span describes an event (an action, occurrence or change).",
        "false": (
            "The marked span is not an event (for example an entity, a static state,"
            " a description or another non-event expression)."
        ),
    },
}
_SCORES = (
    (
        "narrative_gold/setting_concreteness",
        "setting_concreteness_gold",
        CONCRETENESS_Q,
    ),
    (
        "narrative_gold/setting_temporal_grounding",
        "setting_temporal_grounding_gold",
        TEMPORAL_Q,
    ),
)


def _level(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value != int(value) or not 1 <= value <= 5:
        return None
    return int(value) - 1


def setting_candidates(rows: Iterable[dict]) -> Iterator[Candidate]:
    rows = sorted(rows, key=lambda r: r["safe_instance_id"])
    copies = Counter(normalized(row.get("sampled_text") or "") for row in rows)
    for row in rows:
        text = row.get("sampled_text") or ""
        if not text.strip() or copies[normalized(text)] > 1:
            continue
        state = {"passage": text}
        for task, column, question in _SCORES:
            level = _level(row.get(column))
            if level is None or input_chars(state, question) > MAX_INPUT_CHARS:
                continue
            yield Candidate(
                source="narrative_gold",
                task=task,
                source_item_id=row["safe_instance_id"],
                group_id=row["safe_instance_id"],
                balance_label=str(level),
                language="en",
                state=state,
                question=question,
                gold=level,
                overlap_texts=[text],
                date=None,
            )


def _span(value: object) -> tuple[int, int, str] | None:
    try:
        start, end, text = (json.loads(value) if isinstance(value, str) else value)[:3]
    except (TypeError, ValueError):
        return None
    if not isinstance(start, int) or not isinstance(end, int) or start >= end:
        return None
    if not isinstance(text, str) or not text:
        return None
    return start, end, text


def span_candidates(rows: Iterable[dict]) -> Iterator[Candidate]:
    for row in sorted(rows, key=lambda r: r["safe_instance_id"]):
        text = row.get("sampled_text") or ""
        for index in (1, 2):
            gold = row.get(f"span{index}_is_event_gold")
            span = _span(row.get(f"assigned_span{index}"))
            if not isinstance(gold, bool) or span is None:
                continue
            start, end, span_text = span
            if text[start:end] != span_text:
                continue
            state = {
                "passage": f"{text[:start]}<span>{text[start:end]}</span>{text[end:]}",
                "span": span_text,
            }
            if input_chars(state, EVENT_Q) > MAX_INPUT_CHARS:
                continue
            yield Candidate(
                source="narrative_gold",
                task="narrative_gold/span_is_event",
                source_item_id=f"{row['safe_instance_id']}:span{index}",
                group_id=row["safe_instance_id"],
                balance_label=str(gold),
                language="en",
                state=state,
                question=EVENT_Q,
                gold=gold,
                overlap_texts=[text],
                date=None,
            )


def event_candidates(rows: Iterable[dict]) -> Iterator[Candidate]:
    for row in sorted(rows, key=lambda r: r["safe_instance_id"]):
        gold = row.get("causality_rating_gold")
        if gold not in CAUSALITY_Q["criteria"]:
            continue
        if row.get("span1_is_event_gold") is not True:
            continue
        if row.get("span2_is_event_gold") is not True:
            continue
        text = row.get("sampled_text") or ""
        first, second = _span(row.get("assigned_span1")), _span(
            row.get("assigned_span2")
        )
        if first is None or second is None:
            continue
        if not (first[1] <= second[0] or second[1] <= first[0]):
            continue
        if (
            text[first[0] : first[1]] != first[2]
            or text[second[0] : second[1]] != second[2]
        ):
            continue
        marked = text
        for (start, end, _), tag in sorted(
            ((first, "event1"), (second, "event2")), key=lambda item: -item[0][0]
        ):
            marked = f"{marked[:start]}<{tag}>{marked[start:end]}</{tag}>{marked[end:]}"
        state = {"passage": marked, "event_1": first[2], "event_2": second[2]}
        if input_chars(state, CAUSALITY_Q) > MAX_INPUT_CHARS:
            continue
        yield Candidate(
            source="narrative_gold",
            task="narrative_gold/event_causality",
            source_item_id=row["safe_instance_id"],
            group_id=row["safe_instance_id"],
            balance_label=gold,
            language="en",
            state=state,
            question=CAUSALITY_Q,
            gold=gold,
            overlap_texts=[text],
            date=None,
        )


def candidates(root: Path) -> Iterator[Candidate]:
    import pyarrow.parquet as pq

    setting = pq.read_table(
        root / "setting_annotations.parquet", columns=list(SETTING_COLUMNS)
    ).to_pylist()
    events = pq.read_table(
        root / "event_relation_annotations.parquet", columns=list(EVENT_COLUMNS)
    ).to_pylist()
    yield from setting_candidates(setting)
    yield from event_candidates(events)
    yield from span_candidates(events)
