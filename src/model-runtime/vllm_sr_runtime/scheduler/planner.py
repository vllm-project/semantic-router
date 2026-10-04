"""Token-budget micro-batching shared by the profiles."""

from __future__ import annotations


def padded(length: int) -> int:
    return -(-length // 8) * 8


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
