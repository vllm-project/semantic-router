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

from ...plugins.base import AdapterGate, ForwardBatch, ImageInputs
from ...plugins.decisions import RequestPlan
from ..decision3.family import BATCH_SIZE, Decision3Model, Item
from ..decision3.state_once import StateOnceModel
from .gates import Gates


@dataclass(frozen=True)
class GatedItem(Item):
    """A planned question and the gated adapter its row runs (None: every adapter off)."""

    adapter: str | None = None


def adapter_of(item: Item) -> str | None:
    return item.adapter if isinstance(item, GatedItem) else None


class GatedDecisions(Decision3Model):
    gates: Gates

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[Item]:
        if not isinstance(questions, dict):
            return super().plan(state, questions, scan)
        assigned = self.gates.assign(questions)
        plan = super().plan(state, Gates.strip(questions), scan)
        if not assigned:
            return plan
        items = [
            GatedItem(
                **{f.name: getattr(item, f.name) for f in fields(Item)},
                adapter=assigned.get(item.question_id),
            )
            for item in plan.items
        ]
        return replace(plan, items=items)

    def plan_images(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[Any],
        videos: list[Any] | None = None,
    ) -> RequestPlan[Item]:
        stripped = Gates.strip(questions) if isinstance(questions, dict) else questions
        return super().plan_images(state, stripped, images, videos)

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


class TurboDecisions(StateOnceModel):
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

    gates: Gates

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[Item]:
        if not isinstance(questions, dict):
            return super().plan(state, questions, scan)
        plan = super().plan(state, Gates.strip(questions), scan)
        if any(
            isinstance(q, dict) and q.get("type") in BASE_ONLY_TYPES
            for q in questions.values()
        ):
            return plan
        assigned = self.gates.assign(questions)
        replays = [
            GatedItem(
                **{f.name: getattr(item, f.name) for f in fields(Item)}
                | {"state": None, "cost": None},
                adapter=assigned[item.question_id],
            )
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

    def plan_images(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[Any],
        videos: list[Any] | None = None,
    ) -> RequestPlan[Item]:
        stripped = Gates.strip(questions) if isinstance(questions, dict) else questions
        return super().plan_images(state, stripped, images, videos)

    def exact_batches(self, items: list[Item]) -> list[list[int]]:
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
        return self._split(items, super().run)

    def run_shared(
        self, items: list[Item], shared_prefix: int
    ) -> list[list[float] | None]:
        return self._split(
            items, lambda rows: super(TurboDecisions, self).run_shared(rows, 0)
        )

    def run_approximate(self, items: list[Item]) -> list[list[float] | None]:
        return self._split(items, super().run_approximate)

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
