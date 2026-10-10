"""Score-lattice fingerprints of the board's public benchmarks.

A published per-benchmark skill (two decimals) is ``100 (k/N - c)/(1 - c)`` for an integer number
``k`` of correct rows out of ``N``, with ``c`` the mean row chance. Pooled over every entrant and
board snapshot, the published values admit only a narrow interval of chance sums ``S = N c``,
which fingerprints how the board built each benchmark (subset, option counts). ``check`` tests a
built benchmark's ``(N, S)`` against the board; ``feasible_sums`` recovers the intervals.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

TOLERANCE = 0.0051

BOARD_FIXTURE = Path(__file__).parent / "tests" / "data" / "vision-board.json"


def board_values(board: Mapping | None = None) -> dict[str, list[float]]:
    """Distinct published public skills per benchmark, pooled over all snapshots and entries."""
    board = board or json.loads(BOARD_FIXTURE.read_text())
    values: dict[str, set[float]] = {}
    for snap in board["snapshots"]:
        for entry in snap["entries"]:
            for name, cell in entry["bench"].items():
                if cell.get("pub") is not None:
                    values.setdefault(name, set()).add(cell["pub"])
    return {k: sorted(v) for k, v in values.items()}


def nearest_error(n: int, chance_sum: float, value: float) -> float:
    """Distance from ``value`` to the closest displayable skill of an ``(N, S)`` benchmark."""
    c = chance_sum / n
    k = n * (c + value / 100 * (1 - c))
    best = math.inf
    for kk in (math.floor(k), math.ceil(k)):
        if 0 <= kk <= n:
            best = min(best, abs(100 * (kk / n - c) / (1 - c) - value))
    return best


def check(
    n: int, chance_sum: float, values: Iterable[float], tol: float = TOLERANCE
) -> list[float]:
    """Published values that an ``(N, S)`` benchmark cannot produce (empty means consistent)."""
    return [v for v in values if nearest_error(n, chance_sum, v) > tol]


def feasible_sums(
    n: int,
    values: Sequence[float],
    lo: float = 0.0,
    hi: float | None = None,
    step: float = 1e-3,
) -> list[tuple[float, float]]:
    """Intervals of chance sums ``S`` consistent with every value (grid scan in steps of S)."""
    hi = n / 2 if hi is None else hi
    intervals: list[list[float]] = []
    s = lo
    while s <= hi:
        if s > 0 and not check(n, s, values):
            if intervals and s - intervals[-1][1] <= step * 1.5:
                intervals[-1][1] = s
            else:
                intervals.append([s, s])
        s += step
    return [(a, b) for a, b in intervals]


def chance_sum(option_counts: Iterable[int]) -> float:
    return sum(1.0 / n for n in option_counts)
