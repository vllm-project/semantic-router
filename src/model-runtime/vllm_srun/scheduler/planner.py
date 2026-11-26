"""Token-budget micro-batching: the released split of one request and the token costs the batcher counts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..plugins.base import LoadedModel, WorkItem


def padded(length: int) -> int:
    return -(-length // 8) * 8


def cost(item: WorkItem) -> int:
    """Tokens an item costs a forward: its token IDs, or the ``cost`` its family sets (images, audio).

    A family's declared cost can be genuinely zero (a question whose share of
    a shared state and own block both come out to zero), so only an unset
    cost falls back to the token IDs; ``0 or len(item.ids)`` would silently
    replace that declared zero.
    """
    declared: int | None = getattr(item, "cost", None)
    return declared if declared is not None else len(item.ids)


def length_class(length: int) -> int:
    """The power-of-two band of a row's padded length; rows of one band pad little together."""
    return (padded(length) - 1).bit_length()


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
        padded_length = padded(lengths[order[0]])
        if padded_length == 0:
            # A zero cost adds nothing to the budget check below, and every
            # row still in `order` is zero cost too (sorted longest first).
            groups.append(sorted(order))
            break
        rows = budget // padded_length
        groups.append(sorted(order[:rows]))
        order = order[rows:]
    return groups


def exact_split(
    model: LoadedModel[Any, Any] | None, items: list[WorkItem], budget: int | None
) -> list[list[int]]:
    """One request's exact forwards: the model's released split (``exact_batches``), else ``micro_batches``."""
    split = model.exact_batches(items) if model is not None else None
    if split is not None:
        return split
    return micro_batches([cost(item) for item in items], budget)
