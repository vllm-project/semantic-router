"""Shortcut-resistant constructions of data-arms-v2-prereg-2026-09-28.md section 3.

Every function takes normalized source records and returns ``(rows, reason)``
where ``reason`` names why an item produced no rows (``None`` when it did).
Randomness comes only from ``random.Random`` seeded with strings, and choices
are made from lists in deterministic order, so builds do not depend on
``PYTHONHASHSEED``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from v2.data.m2.common import (
    ABSTAIN,
    choice_options,
    make_row,
    noul_options,
    score_options,
    sha,
)
from v2.data.m2.text import join, mentions, units, usable_answer

Rows = list[dict[str, Any]]
Paragraph = tuple[str, str]

REMOVAL_INSTRUCTIONS = (
    "Does the passage state the answer to this question? Question: {}"
)
ANSWERABLE_INSTRUCTIONS = (
    "Do these paragraphs contain enough information to answer this question? "
    "Question: {}"
)
COVERAGE_INSTRUCTIONS = (
    "How much of the information needed to answer this question do these "
    "paragraphs contain? Question: {}"
)
ABSTAIN_INSTRUCTIONS = (
    "Based only on these paragraphs, which option answers the question? Question: {}"
)
RELEVANCE_CHOICE_INSTRUCTIONS = "Which passage answers the question in the state?"
RELEVANCE_NOUL_INSTRUCTIONS = "Does this passage answer the question? Question: {}"
YES_NO = frozenset({"yes", "no"})


def rng_for(seed: str, local_id: str) -> random.Random:
    return random.Random(f"{seed}:{local_id}")


def _clean(text: str) -> str:
    return " ".join(text.split())


def render_paragraphs(paragraphs: Sequence[Paragraph]) -> str:
    return "\n\n".join(
        f"Paragraph {position} — {_clean(title)}: {_clean(text)}"
        for position, (title, text) in enumerate(paragraphs, 1)
    )


def removal_twins(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
) -> tuple[Rows, str | None]:
    """C1: complete vs answer-removed passage, same number of units removed."""
    language, context = record["language"], record["context"]
    answers = [answer for answer in dict.fromkeys(record["answers"]) if answer]
    if not answers or not all(usable_answer(answer) for answer in answers):
        return [], "unusable_answer"
    spans = units(context, language)
    if len(spans) < 4:
        return [], "too_few_units"
    offsets = []
    for answer, start in zip(record["answers"], record.get("answer_starts") or []):
        if (
            isinstance(start, int)
            and start >= 0
            and context[start : start + len(answer)] == answer
        ):
            offsets.append((start, start + len(answer)))
    bearing = [
        index
        for index, (low, high) in enumerate(spans)
        if any(low < end and start < high for start, end in offsets)
        or any(mentions(context[low:high], answer) for answer in answers)
    ]
    if not bearing:
        return [], "answer_not_located"
    count = len(bearing)
    if len(spans) - count < 3 or count > len(spans) // 2:
        return [], "answer_too_widespread"
    bearing_set = frozenset(bearing)
    others = [index for index in range(len(spans)) if index not in bearing_set]
    removed_text = join(
        [context[low:high] for low, high in (spans[i] for i in others)], language
    )
    if any(mentions(removed_text, answer) for answer in answers):
        return [], "answer_survives_removal"
    dropped = frozenset(rng_for(seed, record["local_id"]).sample(others, count))
    complete_text = join(
        [
            context[low:high]
            for index, (low, high) in enumerate(spans)
            if index not in dropped
        ],
        language,
    )
    title = _clean(record.get("title") or "")
    rows = []
    for label, text, twin in (
        (1, complete_text, "complete"),
        (0, removed_text, "removed"),
    ):
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="noul",
                language=language,
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=f"{title}\n\n{text}" if title else text,
                instructions=REMOVAL_INSTRUCTIONS.format(_clean(record["question"])),
                options=noul_options("en"),
                label=label,
                template=template,
                audit={"twin": twin, "units": len(spans), "units_removed": count},
            )
        )
    return rows, None


def relevance(
    record: Mapping[str, Any],
    *,
    source: str,
    choice_family: str,
    noul_family: str,
    namespace: str,
    template: str,
    seed: str,
    negatives: int = 3,
) -> tuple[dict[str, Rows], str | None]:
    """C2: one annotated answer passage vs human-judged non-answer passages."""
    pool = list(record["negatives"])
    if len(pool) < negatives:
        return {}, "too_few_negatives"
    rng = rng_for(seed, record["local_id"])
    chosen = rng.sample(pool, negatives)
    question = _clean(record["question"])
    shared = {
        "source": source,
        "language": record["language"],
        "namespace": namespace,
        "group_key": record["group_key"],
    }
    choice = make_row(
        family=choice_family,
        task_type="choice",
        local_id=f"{record['local_id']}:choice",
        state=f"Question: {question}",
        instructions=RELEVANCE_CHOICE_INSTRUCTIONS,
        options=choice_options([_clean(record["gold"]), *map(_clean, chosen)]),
        label=0,
        template=f"{template}/choice",
        audit={"negatives": negatives},
        rotate_choice=True,
        **shared,
    )
    noul = [
        make_row(
            family=noul_family,
            task_type="noul",
            local_id=f"{record['local_id']}:{twin}",
            state=_clean(passage),
            instructions=RELEVANCE_NOUL_INSTRUCTIONS.format(question),
            options=noul_options("en"),
            label=label,
            template=f"{template}/noul",
            audit={"twin": twin},
            **shared,
        )
        for label, passage, twin in (
            (1, record["gold"], "answer"),
            (0, chosen[0], "non_answer"),
        )
    ]
    return {choice_family: [choice], noul_family: noul}, None


def _answer_paragraphs(gold: Sequence[Paragraph], answer: str) -> list[int]:
    if answer.strip().lower() in YES_NO:
        return []
    return [
        index
        for index, (title, text) in enumerate(gold)
        if mentions(f"{title} {text}", answer)
    ]


def answerability_twins(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    distractors: int,
) -> tuple[Rows, str | None]:
    """C3: all supporting paragraphs vs one removed; each twin has one edit."""
    gold, pool = list(record["gold"]), list(record["distractors"])
    if len(gold) < 2:
        return [], "too_few_supporting"
    if len(pool) < distractors + 1:
        return [], "too_few_distractors"
    answer = record["answer"]
    rng = rng_for(seed, record["local_id"])
    picked = rng.sample(pool, distractors + 1)
    base, extra = picked[:distractors], picked[distractors]
    carriers = _answer_paragraphs(gold, answer)
    removed = rng.choice(carriers) if carriers else rng.randrange(len(gold))
    complete = gold + base[:-1] + [extra]
    incomplete = [p for i, p in enumerate(gold) if i != removed] + base + [extra]
    rng.shuffle(complete)
    rng.shuffle(incomplete)
    incomplete_text = render_paragraphs(incomplete)
    if answer.strip().lower() not in YES_NO and mentions(incomplete_text, answer):
        return [], "answer_in_incomplete_state"
    rows = []
    for label, text, twin in (
        (1, render_paragraphs(complete), "complete"),
        (0, incomplete_text, "removed"),
    ):
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="noul",
                language="en",
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=text,
                instructions=ANSWERABLE_INSTRUCTIONS.format(_clean(record["question"])),
                options=noul_options("en"),
                label=label,
                template=template,
                audit={
                    "twin": twin,
                    "supporting": len(gold),
                    "paragraphs": len(complete),
                    "qtype": record.get("qtype"),
                },
            )
        )
    return rows, None


def coverage_text(present: int, needed: int) -> str:
    if present == 0:
        return f"The paragraphs state none of the {needed} facts needed to answer the question."
    if present == needed:
        return f"The paragraphs state all {needed} facts needed to answer the question."
    return f"The paragraphs state {present} of the {needed} facts needed to answer the question."


def coverage_group(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    padding: int,
) -> tuple[Rows, str | None]:
    """C4: one row per grade g = 0..n with exactly g supporting paragraphs present."""
    gold, pool = list(record["gold"]), list(record["distractors"])
    needed = len(gold)
    if not 2 <= needed <= 9:
        return [], "unsupported_hops"
    total = needed + padding
    if len(pool) < total:
        return [], "too_few_distractors"
    rng = rng_for(seed, record["local_id"])
    options = score_options([coverage_text(g, needed) for g in range(needed + 1)])
    instructions = COVERAGE_INSTRUCTIONS.format(_clean(record["question"]))
    rows = []
    for present in range(needed + 1):
        keep = sorted(rng.sample(range(needed), present))
        paragraphs = [gold[i] for i in keep] + rng.sample(pool, total - present)
        rng.shuffle(paragraphs)
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="score",
                language="en",
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:g{present}",
                state=render_paragraphs(paragraphs),
                instructions=instructions,
                options=options,
                label=present,
                template=template,
                audit={"present": present, "needed": needed, "paragraphs": total},
            )
        )
    return rows, None


def _match_entity(answer: str, candidates: Sequence[str]) -> int | None:
    from v2.data.textnorm import normalize

    target = normalize(answer)
    exact = [i for i, name in enumerate(candidates) if normalize(name) == target]
    if len(exact) == 1:
        return exact[0]
    loose = [
        i
        for i, name in enumerate(candidates)
        if target and (target in normalize(name) or normalize(name) in target)
    ]
    return loose[0] if len(loose) == 1 else None


def abstention_twins(
    record: Mapping[str, Any],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    distractors: int,
) -> tuple[Rows, str | None]:
    """C5: two compared entities plus an abstain option; full evidence vs one
    entity's paragraph removed (gold = abstain)."""
    gold = list(record["gold"])
    if len(gold) != 2:
        return [], "not_two_entities"
    names = [_clean(title) for title, _ in gold]
    if len({name.casefold() for name in names}) != 2:
        return [], "duplicate_entities"
    winner = _match_entity(record["answer"], names)
    if winner is None:
        return [], "answer_not_an_entity"
    pool = list(record["distractors"])
    if len(pool) < distractors + 1:
        return [], "too_few_distractors"
    rng = rng_for(seed, record["local_id"])
    picked = rng.sample(pool, distractors + 1)
    complete = gold + picked[:distractors]
    removed = rng.randrange(2)
    incomplete = [gold[1 - removed]] + picked[: distractors + 1]
    rng.shuffle(complete)
    rng.shuffle(incomplete)
    options = choice_options([names[0], names[1], ABSTAIN])
    rows = []
    for label, paragraphs, twin in (
        (winner, complete, "complete"),
        (2, incomplete, "removed"),
    ):
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="choice",
                language="en",
                namespace=namespace,
                group_key=record["group_key"],
                local_id=f"{record['local_id']}:{twin}",
                state=render_paragraphs(paragraphs),
                instructions=ABSTAIN_INSTRUCTIONS.format(_clean(record["question"])),
                options=options,
                label=label,
                template=template,
                audit={"twin": twin, "qtype": record.get("qtype")},
                rotate_choice=True,
            )
        )
    return rows, None


def numeric_twins(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    family: str,
    namespace: str,
    template: str,
    seed: str,
    instructions: str,
) -> tuple[Rows, dict[str, int]]:
    """C9: each answer is the true candidate of its own problem and the false
    candidate of the next problem with the same digit count (hash order)."""
    buckets: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        buckets.setdefault(len(record["answer"].lstrip("-")), []).append(record)
    rows: Rows = []
    drops = {"singleton_bucket": 0, "equal_partner": 0}
    for size in sorted(buckets):
        members = sorted(buckets[size], key=lambda r: sha(f"{seed}:{r['local_id']}"))
        if len(members) < 2:
            drops["singleton_bucket"] += len(members)
            continue
        for position, record in enumerate(members):
            partner = None
            for step in range(1, len(members)):
                candidate = members[(position + step) % len(members)]
                if candidate["answer"] != record["answer"]:
                    partner = candidate
                    break
            if partner is None:
                drops["equal_partner"] += 1
                continue
            for label, value, twin in (
                (1, record["answer"], "true"),
                (0, partner["answer"], "false"),
            ):
                rows.append(
                    make_row(
                        source=source,
                        family=family,
                        task_type="noul",
                        language="en",
                        namespace=namespace,
                        group_key=record["group_key"],
                        local_id=f"{record['local_id']}:{twin}",
                        state=_clean(record["problem"]),
                        instructions=instructions.format(value),
                        options=noul_options("en"),
                        label=label,
                        template=template,
                        audit={"twin": twin, "digits": size},
                    )
                )
    return rows, drops
