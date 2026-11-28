"""Bucket benchmark receipts separate admission changes from workload changes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bucket_bench as bench


def test_graph_delta_reports_timed_lookups_and_misses_with_the_cache_gauge():
    before = {
        "lookups": 20,
        "misses": 8,
        "captures": 3,
        "replays": 14,
        "eager": 6,
        "full": 0,
        "cached": 3,
    }
    after = {
        "lookups": 30,
        "misses": 12,
        "captures": 4,
        "replays": 21,
        "eager": 9,
        "full": 1,
        "cached": 4,
    }
    assert bench.graph_delta(before, after) == {
        "lookups": 10,
        "misses": 4,
        "captures": 1,
        "replays": 7,
        "eager": 3,
        "full": 1,
        "cached": 4,
    }


def test_graph_delta_accepts_receipts_from_before_lookup_counters():
    assert bench.graph_delta(
        {}, {"captures": 1, "replays": 2, "eager": 3, "full": 4, "cached": 1}
    ) == {
        "lookups": 0,
        "misses": 0,
        "captures": 1,
        "replays": 2,
        "eager": 3,
        "full": 4,
        "cached": 1,
    }
