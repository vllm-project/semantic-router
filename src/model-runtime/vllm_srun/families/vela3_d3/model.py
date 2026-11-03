"""Decisions of the Vela 3.0 d3 sizes: the base's own model with gated adapter passes.

``GatedDecisions`` is the ``decision3`` model with each planned question
marked with the adapter that answers it (``gates.Gates``). A forward whose
rows all run with every adapter off is the base's forward, unchanged. A
forward with gated rows runs once per adapter in the package's order over the
same rows (the batch the base would run), the adapter on for its own rows
only, and each row keeps its own adapter's pass; rows of no adapter keep the
first pass, where every layer returns the base's values for them. Requests
with images run with every adapter off, as the released runtime does, and
rows with an adapter never pack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from ...errors import QuestionError, question_error
from ...plugins.base import AdapterGate, Expired, ForwardBatch, ImageInputs, SurfacePlan
from ...plugins.decisions import RequestPlan, ResolvedDecision
from ..decision3.family import BATCH_SIZE, Decision3Model, Item
from ..decision3.state_once import StateOnceModel
from .gates import Gates
from .spans import (
    Definitions,
    Policy,
    Renderer,
    SpanItem,
    SpanReader,
    invalid,
    is_span,
    plan_question,
)
from .spans import answer as span_answer


@dataclass(frozen=True)
class GatedItem(Item):
    """A planned question and the gated adapter its row runs (None: every adapter off)."""

    adapter: str | None = None


def adapter_of(item: Item) -> str | None:
    return item.adapter if isinstance(item, GatedItem) else None


# ------------------------------------------------------------------ span questions


@dataclass(frozen=True)
class SpanSetup:
    """What a model needs to plan and run span questions (``spans.py``)."""

    policy: Policy
    renderer: Renderer
    reader: SpanReader
    definitions: Definitions


SPAN_IMAGES = "span questions read text only on this model"


def refuse_blank(state: Any) -> None:
    """Refuse a blank text state, as the other Vela families do (``vela2.request.read_state``)."""
    if isinstance(state, str) and not state.strip():
        raise ValueError("state must not be empty or whitespace")


def split_spans(questions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The request's span questions, and the others (the base's)."""
    asked = {k: q for k, q in questions.items() if is_span(q)}
    return asked, {k: q for k, q in questions.items() if k not in asked}


def with_spans(
    plan: RequestPlan[Item] | None,
    state: Any,
    asked: dict[str, Any],
    order: list[str],
    setup: SpanSetup,
    length_error: Callable[[int], QuestionError | None],
    refuse: str | None = None,
) -> RequestPlan[Item]:
    """``plan`` (None: no base question) with the span questions' items, or errors, in request order.

    Every window of a span question goes through the base's ``length_error``:
    only the target part is windowed, so the other parts can still overflow it.
    """
    items: list[Item] = list(plan.items) if plan is not None else []
    errors = dict(plan.errors) if plan is not None else {}
    complete = set(plan.complete_inputs) if plan is not None else set()
    for question_id, question in asked.items():
        try:
            if refuse is not None:
                raise invalid(refuse)
            item = plan_question(
                question_id,
                question,
                state,
                setup.policy,
                setup.renderer,
                setup.definitions,
            )
            refused = length_error(max(len(window.ids) for window in item.windows))
            if refused is not None:
                raise refused
            items.append(item)
        except QuestionError as exc:
            errors[question_id] = question_error("span", exc)
        else:
            if question.get("require_full_input") is True:
                complete.add(question_id)
    if plan is None:
        return RequestPlan(
            question_ids=order,
            items=items,
            errors=errors,
            input_tokens=0,
            complete_inputs=frozenset(complete),
        )
    return replace(
        plan,
        question_ids=order,
        items=items,
        errors=errors,
        complete_inputs=frozenset(complete),
    )


def span_batches(
    items: list[Item], base: Callable[[list[Item]], list[list[int]]]
) -> list[list[int]]:
    """The base's batches of the other items, then each span question alone."""
    rest = [i for i, item in enumerate(items) if not isinstance(item, SpanItem)]
    batches = (
        [[rest[j] for j in batch] for batch in base([items[i] for i in rest])]
        if rest
        else []
    )
    return batches + [[i] for i, item in enumerate(items) if isinstance(item, SpanItem)]


def run_spans(
    items: list[Item],
    base: Callable[[list[Item]], list[Any]],
    setup: SpanSetup,
) -> list[Any]:
    """The other items through ``base``; each span question through its windows and decoder."""
    values: list[Any] = [None] * len(items)
    rest = [i for i, item in enumerate(items) if not isinstance(item, SpanItem)]
    if rest:
        for index, value in zip(rest, base([items[i] for i in rest]), strict=True):
            values[index] = value
    for index, item in enumerate(items):
        if isinstance(item, SpanItem):
            values[index] = span_answer(item, *setup.reader.logits(item))
    return values


def finish_spans(
    response: dict[str, Any], plan: SurfacePlan[Item], results: Any
) -> dict[str, Any]:
    """The span questions' ``spans`` and ``thresholds`` beside their answers."""
    if isinstance(results, Expired):
        return response
    request: RequestPlan[Item] = plan.state
    for index, item in enumerate(request.items):
        if isinstance(item, SpanItem) and item.question_id not in request.errors:
            value = results[index]
            response.setdefault("spans", {})[item.question_id] = value["spans"]
            response.setdefault("thresholds", {})[item.question_id] = value["threshold"]
    return response


def with_adapter(item: Item, adapter: str | None, **changes: Any) -> GatedItem:
    """``item`` as a ``GatedItem`` that runs ``adapter`` (``changes`` replace other fields)."""
    return GatedItem(
        **({f.name: getattr(item, f.name) for f in fields(Item)} | changes),
        adapter=adapter,
    )


class VelaPlanning(Decision3Model):
    """Planning every vela3/3 model shares, over its base's own planning.

    Each entry point (``plan``, ``plan_images``, ``plan_resolved``) refuses a
    blank state, plans the span questions itself and hands the others to the
    base without the package's ``decision_adapter`` key; ``_decisions`` says how
    a size marks its gated rows. Span questions with images, videos or assets
    fail alone (``SPAN_IMAGES``).
    """

    gates: Gates
    span_setup: SpanSetup

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[Item]:
        refuse_blank(state)
        if not isinstance(questions, dict):
            return super().plan(state, questions, scan)
        asked, rest = split_spans(questions)
        plan = self._decisions(state, rest, scan) if rest else None
        return with_spans(
            plan, state, asked, list(questions), self.span_setup, self.length_error
        )

    def _base_plan(
        self, state: Any, questions: dict[str, Any], scan: int | None
    ) -> RequestPlan[Item]:
        """The base's own ``plan`` (the class after this one in the model's MRO)."""
        return super().plan(state, Gates.strip(questions), scan)

    def _decisions(
        self, state: Any, questions: dict[str, Any], scan: int | None
    ) -> RequestPlan[Item]:
        """The base's plan of the non-span questions, each row marked with its gated adapter."""
        assigned = self.gates.assign(questions)
        plan = self._base_plan(state, questions, scan)
        if not assigned:
            return plan
        items = [
            with_adapter(item, assigned.get(item.question_id)) for item in plan.items
        ]
        return replace(plan, items=items)

    def plan_images(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[Any],
        videos: list[Any] | None = None,
    ) -> RequestPlan[Item]:
        refuse_blank(state)
        if not isinstance(questions, dict):
            return super().plan_images(state, questions, images, videos)
        asked, rest = split_spans(questions)
        plan = (
            super().plan_images(state, Gates.strip(rest), images, videos)
            if rest
            else None
        )
        return with_spans(
            plan,
            state,
            asked,
            list(questions),
            self.span_setup,
            self.length_error,
            SPAN_IMAGES,
        )

    def plan_resolved(self, request: ResolvedDecision) -> RequestPlan[Item]:
        """A request with assets or Box / track questions: the base's resolved planning for the
        decisions (every adapter off, as the package answers it), the span questions here.
        """
        refuse_blank(request.state)
        asked, rest = split_spans(request.questions)
        plan = (
            super().plan_resolved(
                replace(
                    request,
                    questions=Gates.strip(rest),
                    uses=[use for use in request.uses if use.question not in asked],
                    targets={k: t for k, t in request.targets.items() if k in rest},
                )
            )
            if rest
            else None
        )
        media = request.images or request.videos or request.uses
        return with_spans(
            plan,
            request.state,
            asked,
            list(request.questions),
            self.span_setup,
            self.length_error,
            SPAN_IMAGES if media else None,
        )

    def answer(self, item: Any, values: Any) -> dict[str, Any]:
        if isinstance(item, SpanItem):
            return {"type": "noul", "noul": values["noul"]}
        return super().answer(item, values)

    def finish_surface(self, plan: SurfacePlan[Item], results: Any) -> dict[str, Any]:
        return finish_spans(super().finish_surface(plan, results), plan, results)


class GatedDecisions(VelaPlanning):
    """vela3-lite, -mini and -flash: the base's own batches, each gated row with its adapter on.

    Span rows run their own noncausal passes, and a batch that holds a gated
    row is not packed.
    """

    def exact_batches(self, items: list[Item]) -> list[list[int]]:
        return span_batches(items, super().exact_batches)

    def run(self, items: list[Item]) -> list[list[float] | None]:
        return run_spans(items, super().run, self.span_setup)

    def run_approximate(self, items: list[Item]) -> list[list[float] | None]:
        return run_spans(items, super().run_approximate, self.span_setup)

    def _packs(self, items: list[Item]) -> bool:
        if any(adapter_of(item) is not None for item in items):
            return False
        return super()._packs(items)

    def _readout(
        self,
        items: list[Item],
        ids: torch.Tensor,
        mask: torch.Tensor,
        last: torch.Tensor,
        images: ImageInputs | None,
        states: list[int] | None = None,
        packed: bool = False,
    ) -> list[list[float] | None]:
        adapters = [adapter_of(item) for item in items]
        order = Gates.pass_order(adapters)
        if not order:
            return super()._readout(items, ids, mask, last, images, states, packed)
        if images is not None or packed:
            raise RuntimeError("rows with a gated adapter run text-only and unpacked")
        out: list[list[float] | None] | None = None
        for name in order:
            rows = tuple(adapter == name for adapter in adapters)
            output = self.engine_model.forward(
                ForwardBatch(
                    input_ids=ids,
                    attention_mask=mask,
                    gather=last[:, None],
                    query=last,
                    lengths=[len(item.ids) for item in items],
                    states=states,
                    adapter=AdapterGate(name, rows),
                )
            )
            probabilities = self._probabilities(items, output.query)
            out = (
                probabilities
                if out is None
                else [
                    new if mine else old
                    for new, old, mine in zip(probabilities, out, rows, strict=True)
                ]
            )
        assert out is not None
        return out


# Question types d3-turbo answers alone (``system_one`` of the released runtime): a request with one runs adapters off.
BASE_ONLY_TYPES = frozenset({"box", "track"})


class TurboDecisions(VelaPlanning, StateOnceModel):
    """vela3-turbo: d3-turbo's state-once answers, then the gated rows again with their adapter on.

    A request's questions run through the base's state-once path with every
    adapter off, the answers plain d3-turbo gives. Each gated question is then
    planned again as a replay row: its full prompt (the same token IDs) under
    the noncausal full-attention mask with its adapter on for the whole pass
    (``decisions.attention`` ``noncausal``), in request order, ``BATCH_SIZE``
    rows per pass, one adapter per pass. A replay comes after the state-once
    rows, so its answer replaces the base's, and its tokens count in
    ``usage.input_tokens``. Requests with images, videos or Box / track
    questions take the base alone.
    """

    def _decisions(
        self, state: Any, questions: dict[str, Any], scan: int | None
    ) -> RequestPlan[Item]:
        plan = self._base_plan(state, questions, scan)
        if any(
            isinstance(q, dict) and q.get("type") in BASE_ONLY_TYPES
            for q in questions.values()
        ):
            return plan
        assigned = self.gates.assign(questions)
        replays = [
            with_adapter(item, assigned[item.question_id], state=None, cost=None)
            for item in plan.items
            if item.question_id in assigned
            and item.media is None
            and item.readouts is None
        ]
        if not replays:
            return plan
        return replace(
            plan,
            items=[*plan.items, *replays],
            input_tokens=plan.input_tokens + sum(len(item.ids) for item in replays),
        )

    def exact_batches(self, items: list[Item]) -> list[list[int]]:
        return span_batches(items, self._decision_batches)

    def _decision_batches(self, items: list[Item]) -> list[list[int]]:
        """The state-once forwards of the request, then each adapter's replays, ``BATCH_SIZE`` per pass."""
        base = [i for i, item in enumerate(items) if adapter_of(item) is None]
        batches = (
            [
                [base[j] for j in batch]
                for batch in super().exact_batches([items[i] for i in base])
            ]
            if base
            else []
        )
        for name in Gates.pass_order(adapter_of(item) for item in items):
            rows = [i for i, item in enumerate(items) if adapter_of(item) == name]
            batches += [
                rows[s : s + BATCH_SIZE] for s in range(0, len(rows), BATCH_SIZE)
            ]
        return batches

    def run(self, items: list[Item]) -> list[list[float] | None]:
        base = super().run
        return run_spans(items, lambda rows: self._split(rows, base), self.span_setup)

    def run_shared(
        self, items: list[Item], shared_prefix: int
    ) -> list[list[float] | None]:
        base = super().run_shared
        return run_spans(
            items,
            lambda rows: self._split(rows, lambda plain: base(plain, 0)),
            self.span_setup,
        )

    def run_approximate(self, items: list[Item]) -> list[list[float] | None]:
        base = super().run_approximate
        return run_spans(items, lambda rows: self._split(rows, base), self.span_setup)

    def shared_context(self, items: list[Item], token_budget: int | None) -> int | None:
        if any(adapter_of(item) is not None for item in items):
            return 0
        return super().shared_context(items, token_budget)

    def _split(
        self,
        items: list[Item],
        base: Callable[[list[Item]], list[list[float] | None]],
    ) -> list[list[float] | None]:
        """State-once rows through ``base``; replays per adapter in passes of ``BATCH_SIZE``."""
        values: list[list[float] | None] = [None] * len(items)
        plain = [i for i, item in enumerate(items) if adapter_of(item) is None]
        if plain:
            for index, value in zip(
                plain, base([items[i] for i in plain]), strict=True
            ):
                values[index] = value
        for name in Gates.pass_order(adapter_of(item) for item in items):
            rows = [i for i, item in enumerate(items) if adapter_of(item) == name]
            for start in range(0, len(rows), BATCH_SIZE):
                chunk = rows[start : start + BATCH_SIZE]
                for index, value in zip(
                    chunk, self._replay([items[i] for i in chunk], name), strict=True
                ):
                    values[index] = value
        return values

    def _replay(self, items: list[Item], name: str) -> list[list[float] | None]:
        ids, mask, last = self._padded(items)
        output = self.engine_model.forward(
            ForwardBatch(
                input_ids=ids,
                attention_mask=mask,
                gather=last[:, None],
                query=last,
                lengths=[len(item.ids) for item in items],
                adapter=AdapterGate(name),
                noncausal=True,
            )
        )
        return self._probabilities(items, output.query)
