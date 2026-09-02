"""Platform-owned bounds and deterministic planning for live capacity evidence."""

from __future__ import annotations

CAPACITY_LOAD_KIND = "closed-loop"
CAPACITY_LOAD_CONFIDENCE_LEVEL = 0.95
MIN_CAPACITY_WARMUP_MULTIPLIER = 2
MAX_CAPACITY_WARMUP_MULTIPLIER = 4
MIN_CAPACITY_MEASUREMENT_REQUESTS = 100
MAX_CAPACITY_MEASUREMENT_REQUESTS = 500
MIN_CAPACITY_REPETITIONS = 3
MAX_CAPACITY_REPETITIONS = 5
MIN_CAPACITY_MEASUREMENT_CLUSTERS_PER_LEVEL = 3
CAPACITY_MAX_ERROR_RATE_CLUSTER_RANGE = 0.05
MAX_CAPACITY_STABILITY_CV = 0.20
MIN_CAPACITY_CONCURRENCY = 2
MAX_CAPACITY_CONCURRENCY = 128


def capacity_concurrency_levels(maximum: int) -> tuple[int, ...]:
    """Return the one admitted geometric load ladder ending at ``maximum``."""

    if not MIN_CAPACITY_CONCURRENCY <= maximum <= MAX_CAPACITY_CONCURRENCY:
        raise ValueError("capacity maximum concurrency must be between 2 and 128")
    levels = [1]
    level = 2
    while level < maximum:
        levels.append(level)
        level *= 2
    levels.append(maximum)
    return tuple(levels)
