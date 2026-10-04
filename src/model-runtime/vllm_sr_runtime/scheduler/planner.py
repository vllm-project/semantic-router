"""Token-budget micro-batching shared by the profiles."""

from __future__ import annotations

from typing import Any


def padded(length: int) -> int:
    return -(-length // 8) * 8


def cost(item: Any) -> int:
    """Tokens an item costs a forward: its token IDs, or the ``cost`` its family sets (images, audio)."""
    return getattr(item, "cost", None) or len(item.ids)


def micro_batches(lengths: list[int], budget: int | None) -> list[list[int]]:
    """Indices per forward: one batch when its padded size fits the budget.

    Otherwise the rows go longest first into batches whose padded size stays
    within the budget, each listing its indices in request order. This is the
    released runtime's split, which keeps batch shapes (and so bits) identical.
    """
    if not lengths:
        return []
    if budget is None or padded(max(lengths)) * len(lengths) <= budget:
        return [list(range(len(lengths)))]
    order = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    if padded(lengths[order[0]]) > budget:
        raise ValueError("a single question exceeds the forward token budget")
    groups: list[list[int]] = []
    while order:
        rows = budget // padded(lengths[order[0]])
        groups.append(sorted(order[:rows]))
        order = order[rows:]
    return groups


def exact_split(model: Any, items: list[Any], budget: int | None) -> list[list[int]]:
    """One request's exact forwards: the model's released split (``exact_batches``), else ``micro_batches``."""
    split = model.exact_batches(items) if model is not None else None
    if split is not None:
        return split
    return micro_batches([cost(item) for item in items], budget)
