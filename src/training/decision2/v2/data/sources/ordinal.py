"""Pure ordinal binning and level balancing for the A6h human Score sources.

Level counts are drawn per row from SHA-256. Cut points are TRAIN quantiles
(linear interpolation between order statistics) or the midpoints between
integer rating anchors. A value within ``guard`` of any cut (inclusive, with a
1e-9 float tolerance) is dropped. Balancing keeps at most ``ratio`` times the
rarest level of a cell per level, where a level without rows counts as zero.
"""

from __future__ import annotations

import bisect
import collections
import itertools
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, TypeVar

from v2.data.sources.common import sha

T = TypeVar("T")
TOLERANCE = 1e-9
BALANCE_RATIO = Fraction(6, 5)


def level_count(family: str, local_id: str, choices: Sequence[int]) -> int:
    return choices[int(sha(f"{family}:{local_id}"), 16) % len(choices)]


def quantile_cuts(values: Sequence[float], levels: int) -> list[float]:
    if levels < 2 or not values:
        raise ValueError("quantile cuts need values and at least two levels")
    ordered = sorted(values)
    last = len(ordered) - 1
    cuts = []
    for k in range(1, levels):
        index, remainder = divmod(k * last, levels)
        low, high = ordered[index], ordered[min(index + 1, last)]
        cuts.append(low + (high - low) * remainder / levels)
    return cuts


def anchor_cuts(top: int) -> list[float]:
    return [anchor + 0.5 for anchor in range(top)]


def anchor_level(value: float, top: int) -> int:
    return min(top, max(0, math.floor(value + 0.5)))


def bin_index(value: float, cuts: Sequence[float]) -> int:
    return bisect.bisect_right(cuts, value)


def near_cut(value: float, cuts: Sequence[float], guard: float) -> bool:
    return any(abs(value - cut) <= guard + TOLERANCE for cut in cuts)


def bands(cuts: Sequence[float], low: float, high: float) -> list[tuple[float, float]]:
    return list(itertools.pairwise([low, *cuts, high]))


def number(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def balance(
    items: Sequence[T],
    *,
    cell: Callable[[T], str],
    level: Callable[[T], int],
    levels: Callable[[T], int],
    ident: Callable[[T], str],
    seed: str,
    ratio: Fraction = BALANCE_RATIO,
) -> tuple[list[T], dict[str, dict[str, Any]]]:
    cells: dict[str, list[T]] = collections.defaultdict(list)
    for item in items:
        cells[cell(item)].append(item)
    kept: set[str] = set()
    report: dict[str, dict[str, Any]] = {}
    for key in sorted(cells):
        members = cells[key]
        sizes = {levels(member) for member in members}
        if len(sizes) != 1:
            raise ValueError(f"cell {key} mixes level counts {sorted(sizes)}")
        size = sizes.pop()
        buckets: list[list[T]] = [[] for _ in range(size)]
        for member in members:
            if not 0 <= level(member) < size:
                raise ValueError(
                    f"cell {key}: level {level(member)} outside 0..{size - 1}"
                )
            buckets[level(member)].append(member)
        limit = min(map(len, buckets)) * ratio.numerator // ratio.denominator
        after = []
        for bucket in buckets:
            chosen = sorted(bucket, key=lambda member: sha(f"{seed}:{ident(member)}"))[
                :limit
            ]
            kept.update(ident(member) for member in chosen)
            after.append(len(chosen))
        report[key] = {
            "before": [len(bucket) for bucket in buckets],
            "after": after,
            "limit": limit,
        }
    return [item for item in items if ident(item) in kept], report
