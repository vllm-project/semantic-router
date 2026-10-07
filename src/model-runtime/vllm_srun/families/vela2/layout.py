"""Layout pieces shared by both Vela 2.0 members: tokenized parts, the truncation order and rows.

A request becomes rows: the first row carries every Choice, Noul, Score and
Set question plus the first Span question, and each further Span question
gets a row of its own (each member reads one span question per sequence).
When a row does not fit the member's input, the part no question reads
shrinks first (to at least 64 tokens, then to one), then the parts the
questions read; a schema that alone does not fit fails its questions with
``max_length_exceeded``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .request import Plan, Question
from .words import Words, words_of

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROMPT_FLOOR = 64


class SchemaTooLongError(ValueError):
    """The questions of a row alone exceed the member's input."""


@dataclass(frozen=True)
class EncodedPart:
    """A part's token IDs; offsets and words only for parts a span question reads."""

    role: str
    ids: NDArray[np.int32]
    offsets: NDArray[np.int32] | None = None
    words: Words | None = None

    def window(self, start: int, end: int, words: Words | None) -> EncodedPart:
        """Tokens ``[start, end)`` of the part, with the words a window reads."""
        offsets = None if self.offsets is None else self.offsets[start:end]
        return EncodedPart(self.role, self.ids[start:end], offsets, words)


@dataclass
class Row:
    """The questions of one sequence (or tree) and the parts they read."""

    questions: list[Question]
    parts: list[EncodedPart]

    @property
    def span(self) -> Question | None:
        return next((q for q in self.questions if q.type == "span"), None)

    @property
    def roles(self) -> list[str]:
        return [part.role for part in self.parts]

    def part(self, role: str) -> EncodedPart:
        return next(part for part in self.parts if part.role == role)

    def read_roles(self) -> set[str]:
        """Every part a question of the row reads (these are protected from shrinking first)."""
        roles: set[str] = set()
        for question in self.questions:
            roles |= set(question.roles)
        return roles

    def shrink_role(self) -> str | None:
        read = self.read_roles()
        return next((role for role in self.roles if role not in read), None)


@dataclass(frozen=True)
class Budget:
    """Kept tokens per part after truncation."""

    lengths: dict[str, int]
    shrink_cut: int = 0
    protected_cut: int = 0


def fit(
    question_lengths: Iterable[int],
    label_lengths: Iterable[int],
    parts: list[tuple[str, int]],
    max_len: int,
    protected: set[str],
    shrink: str | None,
) -> Budget:
    """The packages' truncation order on token costs (``fixed`` counts the markers around the parts)."""
    schema = sum(question_lengths) + sum(label_lengths)
    lengths = dict(parts)
    fixed = 2 + 1 + len(parts)
    shrink_cut = protected_cut = 0

    def over() -> int:
        return fixed + schema + sum(lengths.values()) - max_len

    if shrink is not None:
        if over() > 0:
            keep = max(min(lengths[shrink], PROMPT_FLOOR), lengths[shrink] - over())
            shrink_cut += lengths[shrink] - keep
            lengths[shrink] = keep
        if over() > 0:
            keep = max(1, lengths[shrink] - over())
            shrink_cut += lengths[shrink] - keep
            lengths[shrink] = keep
    if over() > 0:
        for role, _ in parts:
            if role in protected and over() > 0:
                keep = max(1, lengths[role] - over())
                protected_cut += lengths[role] - keep
                lengths[role] = keep
    if over() > 0:
        raise SchemaTooLongError(
            f"the questions alone exceed the input by {over()} tokens"
        )
    return Budget(lengths, shrink_cut, protected_cut)


@dataclass(frozen=True)
class Tokens:
    """How a request is tokenized (no special tokens added).

    ``encode`` returns an encoding with IDs and offsets (parts); ``ids`` the
    IDs of a short string (questions, options, labels, layout pieces), which a
    model may cache across requests.
    """

    encode: Callable[[str], Any]
    ids: Callable[[str], list[int]]

    def part(self, role: str, text: str, with_words: bool) -> EncodedPart:
        """A part's token IDs; with ``with_words`` (a span reads it) also its offsets and words."""
        encoding = self.encode(text)
        ids = np.asarray(encoding.ids, np.int32)
        if not with_words:
            return EncodedPart(role, ids)
        offsets = np.asarray(encoding.offsets, np.int32).reshape(-1, 2)
        return EncodedPart(role, ids, offsets, words_of(text, offsets))


def rows_of(plan: Plan, tokens: Tokens) -> list[Row]:
    """The request's rows; every part is tokenized once and shared by the rows that read it."""
    spans = [q for q in plan.questions if q.type == "span"]
    rest = [q for q in plan.questions if q.type != "span"]
    groups = [rest + spans[:1]] + [[span] for span in spans[1:]]
    span_roles = {span.over for span in spans}
    parts = [
        tokens.part(part.role, part.text, part.role in span_roles)
        for part in plan.state.parts
    ]
    return [Row(questions, parts) for questions in groups if questions]


def word_windows(
    first: NDArray[np.int32], length: int, window: int, stride: int
) -> list[tuple[int, int]]:
    """Token windows ``[a, b)`` over a part that start at word starts and never split a word."""
    starts = sorted({int(x) for x in first})
    if not starts:
        return [(0, length)]
    out, start = [], starts[0]
    while True:
        limit = start + window
        inside = [s for s in starts if start < s < limit]
        end = (
            length
            if limit >= length
            else (max(inside) if inside else min(length, limit))
        )
        out.append((start, end))
        if end >= length:
            break
        later = [s for s in starts if start + stride <= s < end]
        start = later[0] if later else end
    return out


def window_words(words: Words, start: int, end: int) -> tuple[Words, NDArray[np.intp]]:
    """The words whose first token lies in ``[start, end)``, re-based to the window, and their global indices."""
    selected = np.nonzero((words.first >= start) & (words.first < end))[0]
    return Words(words.offsets[selected], words.first[selected] - start), selected
