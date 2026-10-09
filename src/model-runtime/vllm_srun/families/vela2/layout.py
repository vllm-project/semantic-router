"""Layout pieces shared by both Vela 2.0 members: tokenized parts, the truncation order and rows.

A request becomes rows: the first row carries every Choice, Noul, Score and
Set question plus the first Span question, and each further Span question
gets a row of its own (each member reads one span question per sequence).
When a row does not fit the member's input, the part no question reads
shrinks first (to at least 64 tokens, then to one), then the parts the
questions read; a schema that alone does not fit fails its questions with
``max_length_exceeded``.

A part is tokenized only as far as its budgets need (``read_parts`` reads
``need`` tokens of it with ``bounds.read``). A part with more tokens than the
scan budget fails every question that reads it whole with
``scan_budget_exceeded`` (``fail_unscanned``): the model would read it in more
windows than the budget allows, so none of it is scanned. Every other question
reads the part's first tokens, which are all any truncation keeps of it. A
question that truncates (``overflow: truncate``) reads only the part's first
tokens up to its own budget (``EncodedPart.cut``).
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...errors import SCAN_BUDGET_EXCEEDED
from ...text import bounds
from .request import Plan, Question
from .words import Words, words_of

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROMPT_FLOOR = 64


class SchemaTooLongError(ValueError):
    """The questions of a row alone exceed the member's input."""


@dataclass(frozen=True)
class EncodedPart:
    """A part's token IDs; offsets and words only for parts a span question reads.

    ``complete`` is False for a part read only as far as its budgets need:
    ``ids`` are then only its first tokens.
    """

    role: str
    ids: NDArray[np.int32]
    offsets: NDArray[np.int32] | None = None
    words: Words | None = None
    complete: bool = True

    def window(self, start: int, end: int, words: Words | None) -> EncodedPart:
        """Tokens ``[start, end)`` of the part, with the words a window reads."""
        offsets = None if self.offsets is None else self.offsets[start:end]
        return EncodedPart(self.role, self.ids[start:end], offsets, words)

    def cut(self, length: int) -> EncodedPart:
        """The part's first ``length`` tokens, with the words they start."""
        if len(self.ids) <= length:
            return self
        words = None
        if self.words is not None:
            words = window_words(self.words, 0, length)[0]
        return self.window(0, length, words)


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

    @property
    def requires_full_input(self) -> bool:
        return any(question.require_full_input for question in self.questions)

    def require_complete_parts(self) -> None:
        if self.requires_full_input and any(not part.complete for part in self.parts):
            raise SchemaTooLongError("full input exceeds the tokenization budget")

    def require_budget(self, budget: Budget, window_role: str | None = None) -> None:
        """A window may replace one part; every other supplied part must survive."""
        if self.requires_full_input and any(
            part.role != window_role and budget.lengths[part.role] < len(part.ids)
            for part in self.parts
        ):
            raise SchemaTooLongError(
                "full input was clipped to fit the question schema"
            )

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
    model may cache across requests; ``tokenizer`` reads a part's first tokens
    (``bounds.read``).
    """

    encode: Callable[[str], Any]
    ids: Callable[[str], list[int]]
    tokenizer: Any = None

    def part(
        self, role: str, text: str, with_words: bool, need: int | None = None
    ) -> EncodedPart:
        """A part's token IDs; with ``with_words`` (a span reads it) also its offsets and words.

        With ``need``, a part with at least that many tokens is read only that
        far and comes back incomplete, as its first ``need`` tokens.
        """
        count, complete = None, True
        if need is None or self.tokenizer is None:
            encoding = self.encode(text)
        else:
            read = bounds.read(self.tokenizer, text, need)
            encoding = read.encoding
            if read.tokens >= need:
                count, complete = need, False
        ids = np.asarray(encoding.ids[:count], np.int32)
        if not with_words:
            return EncodedPart(role, ids, complete=complete)
        offsets = np.asarray(encoding.offsets[:count], np.int32).reshape(-1, 2)
        end = int(offsets[-1][1]) if count is not None and len(offsets) else len(text)
        words = words_of(text[:end], offsets)
        return EncodedPart(role, ids, offsets, words, complete)


def read_parts(
    plan: Plan, tokens: Tokens, need: int | None = None
) -> list[EncodedPart]:
    """Every part of the state, tokenized once (as far as ``need``) and shared by the rows that read it."""
    span_roles = {q.over for q in plan.questions if q.type == "span"}
    return [
        tokens.part(part.role, part.text, part.role in span_roles, need)
        for part in plan.state.parts
    ]


def fail_unscanned(
    plan: Plan, questions: list[Question], parts: list[EncodedPart], budget: int
) -> list[Question]:
    """``questions`` without those that read a part longer than ``budget``, which
    fail with ``scan_budget_exceeded`` (recorded in ``plan.errors``)."""
    over = {part.role for part in parts if len(part.ids) > budget}
    kept = []
    for question in questions:
        if over.intersection(question.roles) or (question.require_full_input and over):
            plan.errors[question.id] = {
                "type": question.kind,
                "error": SCAN_BUDGET_EXCEEDED,
            }
        else:
            kept.append(question)
    return kept


def rows_for(questions: list[Question], parts: list[EncodedPart]) -> list[Row]:
    """The rows of ``questions`` over ``parts``: one for every question but the
    second and later span questions, which get a row each."""
    spans = [q for q in questions if q.type == "span"]
    rest = [q for q in questions if q.type != "span"]
    groups = [rest + spans[:1]] + [[span] for span in spans[1:]]
    return [Row(group, parts) for group in groups if group]


def rows_of(plan: Plan, tokens: Tokens, need: int | None = None) -> list[Row]:
    """The request's rows, each question reading parts whole; with ``need`` (one
    more than the scan budget), a part over the budget fails the questions that
    read it (``fail_unscanned``)."""
    parts = read_parts(plan, tokens, need)
    questions = plan.questions
    if need is not None:
        questions = fail_unscanned(plan, questions, parts, need - 1)
    return rows_for(questions, parts)


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
        # The last word start strictly inside (start, limit), if any.
        inside = bisect.bisect_left(starts, limit) - 1
        end = (
            length
            if limit >= length
            else (
                starts[inside]
                if inside >= 0 and starts[inside] > start
                else min(length, limit)
            )
        )
        out.append((start, end))
        if end >= length:
            break
        # The first word start in [start + stride, end), if any.
        later = bisect.bisect_left(starts, start + stride)
        start = starts[later] if later < len(starts) and starts[later] < end else end
    return out


def window_words(words: Words, start: int, end: int) -> tuple[Words, NDArray[np.intp]]:
    """The words whose first token lies in ``[start, end)``, re-based to the window, and their global indices."""
    selected = np.nonzero((words.first >= start) & (words.first < end))[0]
    return Words(words.offsets[selected], words.first[selected] - start), selected


def require_window_coverage(
    row: Row, part: EncodedPart, intervals: list[tuple[int, int]]
) -> None:
    """Prove strict windows cover all tokens and each scored word completely.

    Seeing a word's first token is insufficient when the word itself is longer
    than a window: no resulting word score then observes its full text.
    """
    if not row.requires_full_input:
        return
    reached = 0
    for start, end in sorted(intervals):
        if start > reached:
            raise SchemaTooLongError("full input has tokens outside the windows")
        reached = max(reached, end)
    if reached < len(part.ids):
        raise SchemaTooLongError("full input has tokens outside the windows")
    if part.words is None or not len(part.words):
        return
    assert part.offsets is not None
    starts = part.offsets[:, 0]
    if np.any(starts[1:] < starts[:-1]):
        raise SchemaTooLongError("full word coverage cannot be established")
    ends = np.searchsorted(starts, part.words.offsets[:, 1], side="left")
    complete = np.zeros(len(part.words), dtype=bool)
    for start, end in intervals:
        complete |= (part.words.first >= start) & (ends <= end)
    if not np.all(complete):
        raise SchemaTooLongError("full input contains a word split across every window")
