"""The Vela 2.0 0.3B layout: every question of a row in one marker sequence.

``<bos> ([Q] question ([O] option)* [ABS]?)* ([E] label)* [SEP_SCHEMA]
([SEG_role] part)* <eos>``, with markers inserted by token ID so user text
can never create one. A Choice or Score question is read at its ``[O]`` (and
``[ABS]``) markers against the mean of the parts it reads (the pool starts at
the part's ``[SEG_role]`` marker); a Span question contributes only its
``[E]`` labels and is read at the first sub-word of every word of its part.
A labelled part that had to be cut is read in windows of the kept length with
up to 512 tokens of overlap: Choice and Score logits are averaged over the
windows, Set logits take the maximum and word logits are averaged over the
windows that cover the word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .layout import Budget, Row, Tokens, fit, window_words
from .raw import RawRow, RawSpan
from .request import Question
from .words import Words

if TYPE_CHECKING:
    from numpy.typing import NDArray

MARKERS = (
    "[Q]",
    "[O]",
    "[ABS]",
    "[SEP_SCHEMA]",
    "[SEG_user]",
    "[SEG_answer]",
    "[E]",
    "[SEG_context]",
)
UNCOVERED_LOGIT = -30.0


def option_text(name: str, description: str) -> str:
    return name if not description else f"{name}: {description}"


@dataclass
class SequenceQuestion:
    """Where one question sits in a sequence: its ``[Q]``, its ``[O]`` / ``[ABS]`` markers and its pool."""

    question: Question
    query: int
    options: list[int]
    pool: tuple[int, int]


@dataclass
class EncoderSequence:
    """One model input of the 0.3B member (a scheduler work item).

    ``words`` are the positions of the span part's word starts, with their
    code-point offsets and, in a window, their indices among the part's words.
    """

    ids: list[int]
    questions: list[SequenceQuestion]
    labels: list[int] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)
    words: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    word_offsets: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros((0, 2), np.int32)
    )
    word_index: NDArray[np.intp] | None = None
    budget: Budget | None = None


@dataclass(frozen=True)
class _Compiled:
    question: Question
    text: list[int]
    options: list[tuple[str, list[int]]]

    @property
    def length(self) -> int:
        return (
            1
            + len(self.text)
            + sum(1 + len(tokens) for _, tokens in self.options)
            + (1 if self.question.abstain else 0)
        )


class EncoderLayout:
    """Assembles rows into marker sequences for one package's config."""

    def __init__(self, config: dict[str, Any]):
        self.markers: dict[str, int] = dict(config["marker_ids"])
        if set(self.markers) != set(MARKERS):
            raise ValueError(
                "config.json marker_ids must name the eight Vela 2.0 markers"
            )
        self.bos = int(config["bos_token_id"])
        self.eos = int(config["eos_token_id"])
        self.pad = int(config["pad_token_id"])
        self.max_len = int(config["max_length"])
        self.overlap = int(config["window_overlap"])

    def sequences(self, row: Row, tokens: Tokens) -> list[EncoderSequence]:
        """The row as one sequence, or as windows over a labelled part that had to be cut."""
        compiled = [self._compile(q, tokens) for q in row.questions if q.type != "span"]
        span = row.span
        labels = (
            [
                (name, tokens.ids(option_text(name, description)))
                for name, description in span.options
            ]
            if span is not None
            else []
        )
        first = self._assemble(row, compiled, labels)
        assert first.budget is not None
        if not first.budget.protected_cut:
            return [first]
        role = self._window_role(row)
        if role is None:
            return [first]
        part = row.part(role)
        total, kept = len(part.ids), first.budget.lengths[role]
        if kept >= total:
            return [first]
        overlap = min(self.overlap, kept // 4)
        stride = max(1, kept - overlap)
        windows = []
        for start in range(0, max(1, total - overlap), stride):
            end = min(total, start + kept)
            words = index = None
            if part.words is not None:
                words, index = window_words(part.words, start, end)
            window = part.window(start, end, words)
            parts = [window if p.role == role else p for p in row.parts]
            sequence = self._assemble(Row(row.questions, parts), compiled, labels)
            sequence.word_index = index
            windows.append(sequence)
            if end >= total:
                break
        return windows

    def combine(
        self, row: Row, sequences: list[EncoderSequence], outputs: list[Any]
    ) -> RawRow:
        """One row's raw outputs from its sequences' (option logits, span logits)."""
        first = sequences[0]
        raw = RawRow(
            tokens={part.role: len(part.ids) for part in row.parts},
            windows=len(sequences) if len(sequences) > 1 else 0,
            input_tokens=sum(len(sequence.ids) for sequence in sequences),
        )
        for index, entry in enumerate(first.questions):
            stacked: NDArray[np.float32] = np.stack(
                [output[0][index] for output in outputs]
            )
            raw.logits[entry.question.id] = (
                stacked.max(0) if entry.question.type == "set" else stacked.mean(0)
            )
        span = row.span
        if span is None:
            return raw
        if len(sequences) == 1:
            raw.span = RawSpan(
                span.names,
                np.asarray(first.word_offsets),
                outputs[0][1].astype(np.float64),
            )
            return raw
        words = row.part(cast(str, span.over)).words
        assert words is not None
        total = len(words)
        accumulated = np.zeros((total, len(span.names)), np.float64)
        counts = np.zeros(total, np.float64)
        for sequence, (_, logits) in zip(sequences, outputs, strict=True):
            assert sequence.word_index is not None
            covered = sequence.word_index[: len(logits)]
            accumulated[covered] += logits
            counts[covered] += 1
        seen = counts > 0
        raw.span = RawSpan(
            span.names,
            np.asarray(words.offsets)[:total],
            np.where(
                seen[:, None],
                accumulated / np.maximum(counts, 1)[:, None],
                UNCOVERED_LOGIT,
            ),
        )
        return raw

    # -- internals -----------------------------------------------------------

    def _compile(self, question: Question, tokens: Tokens) -> _Compiled:
        options = [
            (name, tokens.ids(option_text(name, description)))
            for name, description in question.options
        ]
        return _Compiled(question, tokens.ids(question.text), options)

    @staticmethod
    def _window_role(row: Row) -> str | None:
        if row.span is not None:
            return cast(str, row.span.over)
        overs = [question.over for question in row.questions]
        if isinstance(overs[0], str) and all(over == overs[0] for over in overs):
            return overs[0]
        return None

    def _assemble(
        self,
        row: Row,
        compiled: list[_Compiled],
        labels: list[tuple[str, list[int]]],
    ) -> EncoderSequence:
        budget = fit(
            (entry.length for entry in compiled),
            (1 + len(tokens) for _, tokens in labels),
            [(part.role, len(part.ids)) for part in row.parts],
            self.max_len,
            row.read_roles(),
            row.shrink_role(),
        )
        markers = self.markers
        ids = [self.bos]
        questions = []
        for entry in compiled:
            query = len(ids)
            ids.append(markers["[Q]"])
            ids.extend(entry.text)
            options = []
            for _, tokens in entry.options:
                options.append(len(ids))
                ids.append(markers["[O]"])
                ids.extend(tokens)
            if entry.question.abstain:
                options.append(len(ids))
                ids.append(markers["[ABS]"])
            questions.append(SequenceQuestion(entry.question, query, options, (0, 0)))
        label_positions = []
        for _, tokens in labels:
            label_positions.append(len(ids))
            ids.append(markers["[E]"])
            ids.extend(tokens)
        ids.append(markers["[SEP_SCHEMA]"])
        ranges: dict[str, tuple[int, int]] = {}
        for part in row.parts:
            start = len(ids)
            ids.append(markers[f"[SEG_{part.role}]"])
            ids.extend(int(token) for token in part.ids[: budget.lengths[part.role]])
            ranges[part.role] = (start, len(ids))
        ids.append(self.eos)
        for placed in questions:
            roles = placed.question.roles
            placed.pool = (ranges[roles[0]][0], ranges[roles[-1]][1])
        sequence = EncoderSequence(ids, questions, budget=budget)
        span = row.span
        if span is not None:
            role = cast(str, span.over)
            part = row.part(role)
            words = (
                part.words
                if part.words is not None
                else Words(np.zeros((0, 2), np.int32), np.zeros(0, np.int32))
            )
            keep = words.first < budget.lengths[role]
            sequence.labels = label_positions
            sequence.label_names = span.names
            sequence.words = (ranges[role][0] + 1 + words.first[keep]).astype(np.int64)
            sequence.word_offsets = words.offsets[keep]
        return sequence


def batch_indices(sequences: list[EncoderSequence]) -> dict[str, NDArray[np.int64]]:
    """The readout index tensors of a padded batch (the published graph's inputs).

    ``q_index`` [row, [Q], pool start, pool end) per question; ``opt_index``
    [row, marker, owning question] per option; ``unit_index`` [row, word
    start]; ``ent_index`` [row, [E]]. An empty index gets one dummy entry so
    no graph input is empty; readouts drop it.
    """
    queries: list[tuple[int, int, int, int]] = []
    options: list[tuple[int, int, int]] = []
    units: list[tuple[int, int]] = []
    labels: list[tuple[int, int]] = []
    for row, sequence in enumerate(sequences):
        for entry in sequence.questions:
            owner = len(queries)
            queries.append((row, entry.query, entry.pool[0], entry.pool[1]))
            options.extend((row, position, owner) for position in entry.options)
        units.extend((row, int(position)) for position in sequence.words)
        labels.extend((row, position) for position in sequence.labels)
    return {
        "q_index": np.asarray(queries or [(0, 0, 0, 1)], np.int64).reshape(-1, 4),
        "opt_index": np.asarray(options or [(0, 0, 0)], np.int64).reshape(-1, 3),
        "unit_index": np.asarray(units or [(0, 0)], np.int64).reshape(-1, 2),
        "ent_index": np.asarray(labels or [(0, 0)], np.int64).reshape(-1, 2),
    }


def split_outputs(
    sequences: list[EncoderSequence],
    option_logits: NDArray[np.float32],
    span_logits: NDArray[np.float32],
) -> list[tuple[list[NDArray[np.float32]], NDArray[np.float32] | None]]:
    """Per sequence: its questions' option logits and its own word x label block."""
    out = []
    option, unit, label = 0, 0, 0
    for sequence in sequences:
        logits = []
        for entry in sequence.questions:
            logits.append(option_logits[option : option + len(entry.options)])
            option += len(entry.options)
        block = None
        if sequence.labels:
            words, names = len(sequence.words), len(sequence.labels)
            block = span_logits[unit : unit + words, label : label + names]
        unit += len(sequence.words)
        label += len(sequence.labels)
        out.append((logits, block))
    return out
