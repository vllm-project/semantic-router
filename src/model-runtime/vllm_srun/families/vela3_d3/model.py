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

from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from ...plugins.base import AdapterGate, ForwardBatch, ImageInputs
from ...plugins.decisions import RequestPlan
from ..decision3.family import Decision3Model, Item
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
