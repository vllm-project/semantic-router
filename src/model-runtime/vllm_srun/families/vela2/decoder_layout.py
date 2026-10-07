"""The Vela 2.0 decoder layout: the parts once, then one block per question (a tree).

The parts are ``Context:\\n`` and, per part, ``<segment role="R">\\n`` TEXT
``\\n</segment>\\n``. A Choice, Noul, Score or Set block is
``\\n\\nTask type: T\\nTarget: R\\nQuestion:\\nQ\\nOptions:``, one
``\\n<option>\\n{"description","key"}\\n</option>`` per option (plus the
abstain option) and the type's suffix; it is read at each option's last token
against the block's last token. A Span block lists its labels the same way
and repeats a target of at most 2,048 tokens after them; its words are read
in the repeat. A longer target is read in windows of at most 1,800 tokens
(stride 1,536, at word starts), each a tree with the span block only, and
word logits are averaged over the windows that cover a word.

Every block attends to the parts and to itself only, so a question's answer
does not depend on the other blocks of its tree. The trees are the packages':
one per row (a span block that windows replace still runs, unread) and one per
window. Every piece is tokenized on its own and the IDs are concatenated, as
the packages do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...systemone import canonical
from .dispatch import Dispatcher, SpanHead
from .layout import Row, SchemaTooLongError, Tokens, fit, window_words, word_windows
from .raw import RawRow, RawSpan
from .request import Question

if TYPE_CHECKING:
    from numpy.typing import NDArray

SUFFIX = {
    "choice": "\n\nSelect the single option best supported by the context and instructions.\nDecision:",
    "score": "\n\nSelect the single option best supported by the context and instructions.\nDecision:",
    "set": "\n\nSelect every option supported by the context and instructions.\nDecision:",
    "span": "\n\nMark every span of the target text that carries one of the labels.\nSpans:",
}
ABSTAIN = ("[ABS]", "None of the shown options applies.")
PARTS_HEAD = "Context:\n"
SEGMENT_CLOSE = "\n</segment>\n"
TEXT_HEAD = "\n\nText:\n"
MIN_PART_BUDGET = 64
UNCOVERED_LOGIT = -30.0


def segment_open(role: str) -> str:
    """The opening line of a part's segment in the packages' decoder prompt."""
    return f'<segment role="{role}">\n'


def option_block(key: str, description: str, tag: str = "option") -> str:
    """One option (or label) of a question block, as canonical JSON between tags."""
    return (
        f"\n<{tag}>\n"
        + canonical({"key": key, "description": description})
        + f"\n</{tag}>"
    )


def question_head(
    kind: str, over: str | tuple[str, ...], text: str, span: bool = False
) -> str:
    """A question block's header: task type, target parts, question and the options' heading."""
    target = ", ".join(over) if isinstance(over, tuple) else over
    return f"\n\nTask type: {kind}\nTarget: {target}\nQuestion:\n{text}\n" + (
        "Labels:" if span else "Options:"
    )


@dataclass
class Block:
    """One question block of a tree.

    ``ends`` are the option endpoints (the abstain option last) or, for a
    span block, the last token of each label block (``starts`` the first);
    ``words`` are the positions of the target's word starts inside the block
    and ``word_index`` their indices among the target part's words.
    """

    question: Question
    ids: list[int]
    ends: list[int]
    head: str = "router"
    starts: list[int] = field(default_factory=list)
    words: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    word_offsets: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros((0, 2), np.int32)
    )
    word_index: NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, np.int64))
    alias: dict[str, str] | None = None
    read: bool = True

    @property
    def query(self) -> int:
        return len(self.ids) - 1

    @property
    def is_span(self) -> bool:
        return self.question.type == "span"


@dataclass
class DecoderTree:
    """Shared parts and the blocks that continue from them (a scheduler work item).

    ``window`` marks a span window's tree, which the packages batch apart from
    the rows' trees.
    """

    prefix: list[int]
    blocks: list[Block] = field(default_factory=list)
    window: bool = False

    @property
    def ids(self) -> list[int]:
        """Every token the tree computes: the parts once, then each block."""
        return self.prefix + [token for block in self.blocks for token in block.ids]


@dataclass
class RowTrees:
    """Where one row's questions are answered: its blocks, its span block or the span's windows.

    ``tokens`` counts the row as the packages render it (a span block that
    windows replace included), for ``usage``.
    """

    row: Row
    tokens: int
    blocks: list[Block] = field(default_factory=list)
    span: Block | None = None
    windows: list[Block] = field(default_factory=list)


class DecoderLayout:
    """Renders rows into trees for one package's config and span-head rule."""

    def __init__(self, config: dict[str, Any], dispatcher: Dispatcher):
        spans = config.get("span_layout") or {}
        self.max_len = int(config["max_length"])
        self.repeat_limit = int(spans.get("hybrid_rmax", 2048))
        self.window = int(spans.get("window", 1800))
        self.stride = int(spans.get("stride", 1536))
        if int(spans.get("window_above", self.repeat_limit)) != self.repeat_limit:
            raise ValueError(
                "span_layout must window exactly the targets it does not repeat"
            )
        self.dispatcher = dispatcher

    def trees(
        self, rows: list[Row], tokens: Tokens
    ) -> tuple[list[DecoderTree], list[RowTrees | None]]:
        """The trees of a request's rows; a row that cannot fit maps to None."""
        trees: list[DecoderTree] = []
        plans: list[RowTrees | None] = []
        for row in rows:
            try:
                plans.append(self._row(row, tokens, trees))
            except SchemaTooLongError:
                plans.append(None)
        return trees, plans

    @staticmethod
    def combine(plan: RowTrees, outputs: dict[int, NDArray[np.float32]]) -> RawRow:
        """A row's raw outputs from its blocks' readouts (keyed by ``id(block)``)."""
        raw = RawRow(
            tokens={part.role: len(part.ids) for part in plan.row.parts},
            input_tokens=plan.tokens,
        )
        for block in plan.blocks:
            raw.logits[block.question.id] = outputs[id(block)]
        span = plan.span
        if span is None:
            return raw
        if not plan.windows:
            logits = np.asarray(outputs[id(span)], np.float64)
            offsets = np.asarray(span.word_offsets)
        else:
            words = plan.row.part(cast(str, span.question.over)).words
            assert words is not None
            accumulated = np.zeros((len(words), len(span.question.names)), np.float64)
            counts = np.zeros(len(words), np.float64)
            for window in plan.windows:
                accumulated[window.word_index] += outputs[id(window)]
                counts[window.word_index] += 1
            seen = counts > 0
            logits = np.where(
                seen[:, None],
                accumulated / np.maximum(counts, 1)[:, None],
                UNCOVERED_LOGIT,
            )
            offsets = np.asarray(words.offsets)
            raw.windows = len(plan.windows)
        raw.span = RawSpan(
            span.question.names, offsets, logits, head=span.head, alias=span.alias
        )
        return raw

    # -- internals -----------------------------------------------------------

    def _row(self, row: Row, tokens: Tokens, trees: list[DecoderTree]) -> RowTrees:
        span = row.span
        routed = self.dispatcher.route(span) if span is not None else None
        questions = [
            cast(SpanHead, routed).question if q is span else q for q in row.questions
        ]
        prefix, blocks, span_block, rendered = self._render(
            row, questions, tokens, routed
        )
        plan = RowTrees(
            row, rendered, blocks=[b for b in blocks if not b.is_span], span=span_block
        )
        target = row.part(cast(str, span.over)) if span is not None else None
        if target is None or len(target.ids) <= self.repeat_limit:
            trees.append(DecoderTree(prefix, blocks))
            return plan
        assert routed is not None and span_block is not None
        assert target.words is not None
        span_block.read = False
        trees.append(DecoderTree(prefix, [*plan.blocks, span_block]))
        position = row.parts.index(target)
        for start, end in word_windows(
            target.words.first, len(target.ids), self.window, self.stride
        ):
            words, selected = window_words(target.words, start, end)
            parts = list(row.parts)
            parts[position] = target.window(start, end, words)
            window_prefix, window_blocks, window_span, window_tokens = self._render(
                Row([routed.question], parts), [routed.question], tokens, routed
            )
            assert window_span is not None
            window_span.word_index = selected[: len(window_span.words)]
            plan.windows.append(window_span)
            plan.tokens += window_tokens
            trees.append(DecoderTree(window_prefix, window_blocks, window=True))
        return plan

    def _render(
        self,
        row: Row,
        questions: list[Question],
        tokens: Tokens,
        routed: SpanHead | None,
    ) -> tuple[list[int], list[Block], Block | None, int]:
        """The packages' render of one row: parts, blocks to run, the span block and the token count.

        A span block whose target is too long to repeat is returned but not
        among the blocks to run (windows replace it).
        """
        ids = tokens.ids
        compiled = []
        span_question: Question | None = None
        labels: list[list[int]] = []
        for question in questions:
            if question.type == "span":
                span_question = question
                labels = [ids(option_block(n, d, "label")) for n, d in question.options]
                continue
            options = [ids(option_block(n, d)) for n, d in question.options]
            head = ids(question_head(question.type, question.over, question.text))
            suffix = ids(SUFFIX[question.type])
            abstain = ids(option_block(*ABSTAIN)) if question.abstain else []
            length = 1 + len(head) + len(suffix) + len(abstain)
            length += sum(1 + len(option) for option in options) + int(question.abstain)
            compiled.append((question, head, options, abstain, suffix, length))
        overhead = len(ids(PARTS_HEAD)) + sum(
            len(ids(segment_open(role))) + len(ids(SEGMENT_CLOSE)) for role in row.roles
        )
        repeat = False
        if span_question is not None:
            span_head = ids(
                question_head("span", span_question.over, span_question.text, True)
            )
            span_suffix = ids(SUFFIX["span"])
            overhead += len(span_head) + len(span_suffix)
            span_role = cast(str, span_question.over)
            target = len(row.part(span_role).ids)
            repeat = target <= self.repeat_limit
            if repeat:
                overhead += len(ids(TEXT_HEAD)) + len(ids(segment_open(span_role)))
                overhead += len(ids(SEGMENT_CLOSE))
                overhead += min(target, max(0, self.max_len - overhead) // 2)
        budget = fit(
            (entry[-1] for entry in compiled),
            (1 + len(label) for label in labels),
            [(part.role, len(part.ids)) for part in row.parts],
            max(MIN_PART_BUDGET, self.max_len - overhead),
            row.read_roles(),
            row.shrink_role(),
        )
        prefix = list(ids(PARTS_HEAD))
        for part in row.parts:
            prefix += ids(segment_open(part.role))
            prefix += [int(token) for token in part.ids[: budget.lengths[part.role]]]
            prefix += ids(SEGMENT_CLOSE)
        blocks = []
        for question, head, options, abstain, suffix, _ in compiled:
            block_ids, ends = list(head), []
            for option in options + ([abstain] if question.abstain else []):
                block_ids += option
                ends.append(len(block_ids) - 1)
            blocks.append(Block(question, block_ids + suffix, ends))
        rendered = len(prefix) + sum(len(block.ids) for block in blocks)
        if span_question is None or not labels:
            return prefix, blocks, None, rendered
        block_ids, starts, ends = list(span_head), [], []
        for label in labels:
            starts.append(len(block_ids))
            block_ids += label
            ends.append(len(block_ids) - 1)
        assert routed is not None
        span_role = cast(str, span_question.over)
        part = row.part(span_role)
        assert part.words is not None
        kept = budget.lengths[span_role]
        keep = part.words.first < kept
        words: NDArray[np.int64] = np.zeros(0, np.int64)
        if repeat:
            block_ids += ids(TEXT_HEAD) + ids(segment_open(span_role))
            words = (len(block_ids) + part.words.first[keep]).astype(np.int64)
            block_ids += [int(token) for token in part.ids[:kept]] + ids(SEGMENT_CLOSE)
        span_block = Block(
            span_question,
            block_ids + span_suffix,
            ends,
            head=routed.head,
            starts=starts,
            words=words,
            word_offsets=part.words.offsets[keep],
            word_index=np.nonzero(keep)[0],
            alias=routed.alias,
        )
        if repeat:
            blocks.append(span_block)
        return prefix, blocks, span_block, rendered + len(span_block.ids)
