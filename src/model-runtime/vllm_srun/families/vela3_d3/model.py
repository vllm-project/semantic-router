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
from typing import Any, ClassVar, Literal, cast

import torch

from ...errors import QuestionError, question_error
from ...plugins.base import AdapterGate, Expired, ImageInputs, SurfacePlan
from ...plugins.decisions import RequestPlan, ResolvedDecision
from ..decision3.family import BATCH_SIZE, Decision3Model, Item
from ..decision3.state_once import StateOnceModel
from .gates import Gates
from .span_policy import Definitions, Policy
from .span_read import SpanReader
from .span_read import answer as span_answer
from .span_render import Renderer
from .spans import SpanItem, invalid, is_span, plan_question


@dataclass(frozen=True)
class GatedItem(Item):
    """A planned question and the gated adapter its row runs (None: every adapter off)."""

    adapter: str | None = None


def adapter_of(item: Item) -> str | None:
    return item.adapter if isinstance(item, GatedItem) else None


# ------------------------------------------------------------------ span questions


@dataclass(frozen=True)
class SpanSetup:
    """What a model needs to plan and run span questions (``spans.py`` and its ``span_*`` modules)."""

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
    asked: dict[str, Any],
    order: list[str],
    plan_span: Callable[[str, dict[str, Any]], SpanItem],
) -> RequestPlan[Item]:
    """``plan`` (None: no base question) with the span questions' items, or errors, in request order."""
    items: list[Item] = list(plan.items) if plan is not None else []
    errors = dict(plan.errors) if plan is not None else {}
    complete = set(plan.complete_inputs) if plan is not None else set()
    for question_id, question in asked.items():
        try:
            items.append(plan_span(question_id, question))
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


def refuse_with_media(question_id: str, question: dict[str, Any]) -> SpanItem:
    """Span questions of a request with images, videos or assets: they read text only."""
    raise invalid(SPAN_IMAGES)


@dataclass(frozen=True)
class Lane:
    """Which way a group of items runs: the base's batches, one adapter's noncausal replays, or span passes."""

    kind: Literal["base", "replay", "span"]
    adapter: str | None = None


BASE, SPAN = Lane("base"), Lane("span")


def lanes(items: list[Item], replays: bool) -> list[tuple[Lane, list[int]]]:
    """The items' indices by lane, in the order their forwards run.

    The base lane comes first, then (``replays``: vela3-turbo) each adapter's
    replays in the package's pass order, then the span questions. Without
    replays a gated row stays in the base lane, which gates it in its forward.
    """

    def lane_of(item: Item) -> Lane:
        if isinstance(item, SpanItem):
            return SPAN
        adapter = adapter_of(item)
        return Lane("replay", adapter) if replays and adapter is not None else BASE

    keys = [lane_of(item) for item in items]
    order = [
        BASE,
        *(Lane("replay", name) for name in Gates.pass_order(k.adapter for k in keys)),
        SPAN,
    ]
    groups = [
        (lane, [i for i, key in enumerate(keys) if key == lane]) for lane in order
    ]
    return [(lane, rows) for lane, rows in groups if rows]


def chunks(count: int) -> list[list[int]]:
    """``range(count)`` in passes of ``BATCH_SIZE``."""
    return [
        list(range(s, min(s + BATCH_SIZE, count))) for s in range(0, count, BATCH_SIZE)
    ]


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


class Vela3D3Model(Decision3Model):
    """What every vela3/3 model adds to its base model (``Decision3Model`` or ``StateOnceModel``).

    The base model is whatever follows this class in a size's MRO, so every
    ``super()`` call here reaches the base. Each planning entry point
    (``plan``, ``plan_images``, ``plan_resolved``) refuses a blank state, plans
    the span questions itself and hands the others to the base without the
    package's ``decision_adapter`` key; ``_decisions`` says how a size marks its
    gated rows. Span questions with images, videos or assets fail alone
    (``SPAN_IMAGES``). Running splits the items into ``lanes``; ``replays``
    says whether gated rows run as their own noncausal replays.
    """

    gates: Gates
    span_setup: SpanSetup
    replays: ClassVar[bool] = False

    def attach(self, gates: Gates, span_setup: SpanSetup) -> None:
        """Give the loaded base model the package's gates and span setup (``Vela3D3Family.load``)."""
        self.gates = gates
        self.span_setup = span_setup

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[Item]:
        refuse_blank(state)
        if not isinstance(questions, dict):
            return super().plan(state, questions, scan)
        asked, rest = split_spans(questions)
        plan = self._decisions(state, rest, scan) if rest else None
        return with_spans(plan, asked, list(questions), self._span_planner(state))

    def _span_planner(self, state: Any) -> Callable[[str, dict[str, Any]], SpanItem]:
        """Plans one span question over ``state`` with the package's span setup and the base's length check."""
        setup = self.span_setup

        def plan_span(question_id: str, question: dict[str, Any]) -> SpanItem:
            return plan_question(
                question_id,
                question,
                state,
                setup.policy,
                setup.renderer,
                setup.definitions,
                self.length_error,
            )

        return plan_span

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
        return with_spans(plan, asked, list(questions), refuse_with_media)

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
            asked,
            list(request.questions),
            refuse_with_media if media else self._span_planner(request.state),
        )

    def exact_batches(self, items: list[Item]) -> list[list[int]]:
        batches: list[list[int]] = []
        for lane, rows in lanes(items, self.replays):
            if lane == BASE:
                local = super().exact_batches([items[i] for i in rows])
            elif lane == SPAN:
                local = [[j] for j in range(len(rows))]
            else:
                local = chunks(len(rows))
            batches += [[rows[j] for j in batch] for batch in local]
        return batches

    def run(self, items: list[Item]) -> list[list[float] | None]:
        return self._run_lanes(items, super().run)

    def run_approximate(self, items: list[Item]) -> list[list[float] | None]:
        return self._run_lanes(items, super().run_approximate)

    def _run_lanes(
        self, items: list[Item], base: Callable[[list[Item]], list[Any]]
    ) -> list[Any]:
        """Each lane's items through ``base``, their replays or their span passes, in ``lanes`` order."""
        values: list[Any] = [None] * len(items)
        for lane, rows in lanes(items, self.replays):
            group = [items[i] for i in rows]
            if lane == BASE:
                found = base(group)
            elif lane == SPAN:
                reader = self.span_setup.reader
                found = [
                    span_answer(item, *reader.logits(item))
                    for item in cast(list[SpanItem], group)
                ]
            else:
                found = [
                    value
                    for chunk in chunks(len(group))
                    for value in self._replay([group[j] for j in chunk], lane.adapter)
                ]
            for index, value in zip(rows, found, strict=True):
                values[index] = value
        return values

    def _replay(
        self, items: list[Item], adapter: str | None
    ) -> list[list[float] | None]:
        """One pass of full prompts under the noncausal mask, ``adapter`` on for every row."""
        assert adapter is not None
        ids, mask, last = self._padded(items)
        return Decision3Model._readout(
            self,
            items,
            ids,
            mask,
            last,
            None,
            adapter=AdapterGate(adapter),
            noncausal=True,
        )

    def answer(self, item: Any, values: Any) -> dict[str, Any]:
        if isinstance(item, SpanItem):
            return {"type": "noul", "noul": values["noul"]}
        return super().answer(item, values)

    def finish_surface(self, plan: SurfacePlan[Item], results: Any) -> dict[str, Any]:
        return finish_spans(super().finish_surface(plan, results), plan, results)


class GatedDecisions(Vela3D3Model):
    """vela3-lite, -mini and -flash: the base's own batches, each gated row with its adapter on.

    A batch with gated rows runs once per adapter in pass order over the same
    rows, each row keeping its own adapter's pass, and is never packed.
    """

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
        adapter: AdapterGate | None = None,
        noncausal: bool = False,
    ) -> list[list[float] | None]:
        adapters = [adapter_of(item) for item in items]
        order = Gates.pass_order(adapters)
        if not order or adapter is not None or noncausal:
            return super()._readout(
                items, ids, mask, last, images, states, packed, adapter, noncausal
            )
        if images is not None or packed:
            raise RuntimeError("rows with a gated adapter run text-only and unpacked")
        out: list[list[float] | None] | None = None
        for name in order:
            rows = tuple(adapter == name for adapter in adapters)
            probabilities = super()._readout(
                items, ids, mask, last, None, states, adapter=AdapterGate(name, rows)
            )
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


class TurboDecisions(Vela3D3Model, StateOnceModel):
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

    replays = True

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

    def run_shared(
        self, items: list[Item], shared_prefix: int
    ) -> list[list[float] | None]:
        base = super().run_shared
        return self._run_lanes(items, lambda group: base(group, shared_prefix))

    def shared_context(self, items: list[Item], token_budget: int | None) -> int | None:
        if any(adapter_of(item) is not None for item in items):
            return 0
        return super().shared_context(items, token_budget)
