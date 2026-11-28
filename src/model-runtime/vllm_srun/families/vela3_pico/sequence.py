"""The pico's one pass: every question of a request in one marker sequence (``vela3_schema._Core.one_pass``).

``<bos> ([Q] question ([O] option)* [ABS]?)* ([E] label)* [SEP_SCHEMA]
([SEG_role] part)* <eos>``: the Choice and Set questions in request order,
then each span question's ``[E]`` labels as one group, in request order, then
the state's parts. A decision is pooled over every part, a router question
over the part (or parts) it reads, and a span group's labels read the first
sub-word of every word of its own target part. Markers are inserted by token
ID, so text never creates one.

The package reads a request in one pass only when nothing has to be cut: the
truncation order of ``vela2.layout.fit`` (the package's ``assemble``) must keep
every token of every part at the window (8,192 tokens). Otherwise, or when the
questions alone do not fit, ``build`` returns None and the request is answered
question by question. A pass reads its parts whole, so the target tokens of a
span group are its part's full token count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..vela2.encoder_layout import EncoderSequence, SequenceQuestion, option_text
from ..vela2.layout import EncodedPart, SchemaTooLongError, Tokens, fit
from .request import DECISION, SPAN, Asked

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Group:
    """One span question's ``[E]`` group: where its words and labels sit among the sequence's."""

    asked: Asked
    words: tuple[int, int]
    labels: tuple[int, int]
    offsets: NDArray[np.int32]
    role: str
    target_tokens: int


@dataclass(frozen=True)
class OnePass:
    """The request's one sequence and its span groups."""

    sequence: EncoderSequence
    groups: tuple[Group, ...]


@dataclass(frozen=True)
class Layout:
    """The marker IDs and special tokens of a package's ``config.json``, and the window of one pass."""

    markers: dict[str, int]
    bos: int
    eos: int
    max_len: int


def _block(asked: Asked, tokens: Tokens) -> tuple[list[int], list[list[int]], bool]:
    question = asked.question
    options = [tokens.ids(option_text(name, desc)) for name, desc in question.options]
    return tokens.ids(question.text), options, question.abstain


def build(
    asked: Sequence[Asked],
    parts: Sequence[EncodedPart],
    tokens: Tokens,
    layout: Layout,
) -> OnePass | None:
    """The request's one sequence, or None when it does not fit uncut."""
    roles = [part.role for part in parts]
    blocks = [(a, *_block(a, tokens)) for a in asked if a.path != SPAN]
    groups = [
        (
            a,
            [tokens.ids(option_text(name, desc)) for name, desc in a.question.options],
        )
        for a in asked
        if a.path == SPAN
    ]
    read: set[str] = set()
    for a in asked:
        read |= set(roles) if a.path == DECISION else set(a.question.roles)
    try:
        budget = fit(
            (
                1 + len(text) + sum(1 + len(o) for o in options) + int(abstain)
                for _, text, options, abstain in blocks
            ),
            (1 + len(label) for _, group in groups for label in group),
            [(part.role, len(part.ids)) for part in parts],
            layout.max_len,
            read,
            next((role for role in roles if role not in read), None),
        )
    except SchemaTooLongError:
        return None
    if budget.shrink_cut or budget.protected_cut:
        return None
    markers = layout.markers
    ids = [layout.bos]
    placed: list[tuple[Asked, int, list[int]]] = []
    for a, text, options, abstain in blocks:
        query = len(ids)
        ids += [markers["[Q]"], *text]
        positions = []
        for option in options:
            positions.append(len(ids))
            ids += [markers["[O]"], *option]
        if abstain:
            positions.append(len(ids))
            ids.append(markers["[ABS]"])
        placed.append((a, query, positions))
    label_positions: list[list[int]] = []
    for _, group_labels in groups:
        here = []
        for label in group_labels:
            here.append(len(ids))
            ids += [markers["[E]"], *label]
        label_positions.append(here)
    ids.append(markers["[SEP_SCHEMA]"])
    ranges: dict[str, tuple[int, int]] = {}
    for part in parts:
        start = len(ids)
        ids.append(markers[f"[SEG_{part.role}]"])
        ids.extend(int(token) for token in part.ids)
        ranges[part.role] = (start, len(ids))
    ids.append(layout.eos)
    questions = []
    for a, query, positions in placed:
        over = roles if a.path == DECISION else list(a.question.roles)
        pool = (ranges[over[0]][0], ranges[over[-1]][1])
        questions.append(SequenceQuestion(a.question, query, positions, pool))
    words: list[NDArray[np.int64]] = []
    offsets: list[NDArray[np.int32]] = []
    entities: list[int] = []
    names: list[str] = []
    placed_groups = []
    unit = 0
    for (a, _), here in zip(groups, label_positions, strict=True):
        role = str(a.question.over)
        part = next(p for p in parts if p.role == role)
        assert part.words is not None
        starts = (ranges[role][0] + 1 + part.words.first).astype(np.int64)
        words.append(starts)
        offsets.append(part.words.offsets)
        placed_groups.append(
            Group(
                a,
                (unit, unit + len(starts)),
                (len(entities), len(entities) + len(here)),
                part.words.offsets,
                role,
                len(part.ids),
            )
        )
        unit += len(starts)
        entities += here
        names += a.question.names
    sequence = EncoderSequence(ids, questions)
    if placed_groups:
        sequence.labels = entities
        sequence.label_names = names
        sequence.words = np.concatenate(words)
        sequence.word_offsets = np.concatenate(offsets)
    return OnePass(sequence, tuple(placed_groups))
