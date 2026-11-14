"""A pico request question by question: what ``system_one`` does when the request is not one pass.

Decisions run on the decision layout, one sequence each,
``[BOS] [Q] instructions ([O] option)* [SEP_SCHEMA] [SEG_user] state [EOS]``,
the whole state as one part (a string as it is, anything else as compact
JSON), batched together by length (``DecisionEncoder.predict``). Router
questions run in the Vela 2.0 layout, one row per part they read (and per
``overflow``), each row its own call, read in windows past the window size.
Each span question is a row and a call of its own; an evidence question reads
one window of up to ``span.evidence.max_len`` tokens. Every call runs on the
package's one network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...errors import MAX_LENGTH_EXCEEDED
from ..vela2.encoder_layout import (
    EncoderLayout,
    EncoderSequence,
    SequenceQuestion,
    option_text,
)
from ..vela2.layout import Row, SchemaTooLongError, Tokens, fail_unscanned, read_parts
from ..vela2.request import Plan as PartsPlan
from ..vela3_calls import DECISION_BUDGET, Call, Vela3Item
from ..vela3_shared import text
from .request import DECISION, ROUTER, Asked, Read

SPAN_BUDGET = 32768
SPAN_ROWS = 32


@dataclass
class Placed:
    """Where one row's inputs sit among a request's items."""

    row: Row
    layout: EncoderLayout
    items: list[int]


@dataclass
class Fallback:
    """A request planned question by question: where each question's inputs sit."""

    decisions: dict[str, int] = field(default_factory=dict)
    rows: dict[str, Placed] = field(default_factory=dict)
    complete: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Layouts:
    """A package's decision layout and its router and evidence windows."""

    markers: dict[str, int]
    bos: int
    eos: int
    max_len: int
    window: EncoderLayout
    evidence: EncoderLayout


def plan(
    request: Read,
    tokens: Tokens,
    layouts: Layouts,
    scan: int,
    items: list[Vela3Item],
    errors: dict[str, dict[str, Any]],
) -> Fallback:
    """Plan ``request``'s valid questions question by question, appending their inputs to ``items``."""
    out = Fallback()
    _decisions(request, tokens, layouts, items, errors, out)
    _rows(request, tokens, layouts, scan, items, errors, out)
    return out


def _decisions(
    request: Read,
    tokens: Tokens,
    layouts: Layouts,
    items: list[Vela3Item],
    errors: dict[str, dict[str, Any]],
    out: Fallback,
) -> None:
    asked = [a for a in request.asked if a.path == DECISION]
    if not asked:
        return
    state = tokens.ids(text("" if request.raw_state is None else request.raw_state))
    call = Call(None, DECISION_BUDGET)
    for entry in asked:
        sequence = _decision(entry, state, tokens, layouts)
        if sequence is None:
            errors[entry.question.id] = {
                "type": entry.question.kind,
                "error": MAX_LENGTH_EXCEEDED,
            }
            continue
        out.decisions[entry.question.id] = len(items)
        if entry.question.require_full_input:
            out.complete.add(entry.question.id)
        items.append(Vela3Item(sequence, call))


def _decision(
    asked: Asked, state: list[int], tokens: Tokens, layouts: Layouts
) -> EncoderSequence | None:
    """``DecisionEncoder.encode``: the question as rendered, then the whole state; None when it does not fit."""
    question, markers = asked.question, layouts.markers
    head = [layouts.bos, markers["[Q]"], *tokens.ids(question.text)]
    options = []
    for name, description in question.options:
        options.append(len(head))
        head += [markers["[O]"], *tokens.ids(option_text(name, description))]
    head.append(markers["[SEP_SCHEMA]"])
    over = len(head) + len(state) + 2 - layouts.max_len
    if over > 0:
        if not question.truncate or over > len(state):
            return None
        state = state[: len(state) - over]
    start = len(head)
    ids = [*head, markers["[SEG_user]"], *state]
    end = len(ids)
    ids.append(layouts.eos)
    return EncoderSequence(ids, [SequenceQuestion(question, 1, options, (start, end))])


def _rows(
    request: Read,
    tokens: Tokens,
    layouts: Layouts,
    scan: int,
    items: list[Vela3Item],
    errors: dict[str, dict[str, Any]],
    out: Fallback,
) -> None:
    asked = [a for a in request.asked if a.path != DECISION]
    if not asked or request.state is None:
        return
    window = layouts.window.max_len
    budget = max(scan, window)
    cut = max(
        (_layout_of(a, layouts).max_len for a in asked if a.question.truncate),
        default=0,
    )
    parts_plan = PartsPlan(
        request.state, request.question_ids, [a.question for a in asked]
    )
    parts = read_parts(parts_plan, tokens, max(budget, cut) + 1)
    whole = fail_unscanned(
        parts_plan,
        [a.question for a in asked if not a.question.truncate],
        parts,
        budget,
    )
    errors.update(parts_plan.errors)
    kept = {question.id for question in whole} | {
        a.question.id for a in asked if a.question.truncate
    }
    groups: dict[tuple[Any, ...], list[Asked]] = {}
    for entry in asked:
        question = entry.question
        if question.id not in kept:
            continue
        if entry.path == ROUTER:
            key: tuple[Any, ...] = (ROUTER, question.over, question.truncate)
        else:
            key = ("span", question.id)
        groups.setdefault(key, []).append(entry)
    turns: dict[int, int] = {}
    for members in groups.values():
        first = members[0]
        layout = _layout_of(first, layouts)
        row = Row([entry.question for entry in members], list(parts))
        if first.question.truncate:
            # Only the parts the row reads get its layout's window.
            read = row.read_roles()
            row.parts = [
                part.cut(layout.max_len if part.role in read else window)
                for part in parts
            ]
        call_budget = (
            SPAN_BUDGET if first.path == ROUTER else max(layout.max_len, SPAN_BUDGET)
        )
        try:
            sequences = layout.sequences(
                row, tokens, in_windows=not first.question.truncate
            )
        except SchemaTooLongError:
            for question in row.questions:
                errors[question.id] = {
                    "type": question.kind,
                    "error": MAX_LENGTH_EXCEEDED,
                }
            continue
        turn = turns.get(call_budget, 0)
        turns[call_budget] = turn + 1
        call = Call(None, call_budget, SPAN_ROWS, turn)
        placed = Placed(
            row, layout, list(range(len(items), len(items) + len(sequences)))
        )
        items.extend(Vela3Item(sequence, call) for sequence in sequences)
        for question in row.questions:
            out.rows[question.id] = placed
            if question.require_full_input:
                out.complete.add(question.id)


def _layout_of(asked: Asked, layouts: Layouts) -> EncoderLayout:
    """Evidence span questions read one long window; every other row the router window."""
    if asked.route is not None and asked.route.evidence:
        return layouts.evidence
    return layouts.window
