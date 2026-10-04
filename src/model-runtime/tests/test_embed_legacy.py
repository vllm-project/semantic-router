"""The embed A/B tool's confidence intervals (``tools/embed_legacy.py``)."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "embed_legacy.py"


@pytest.fixture(scope="module")
def el():
    spec = importlib.util.spec_from_file_location("embed_legacy", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_intervals_bound_runtime_minus_legacy(el):
    legacy = [int((10 + i % 7) * 1e6) for i in range(60)]
    faster = {
        "legacy": legacy,
        "runtime": [ns - 2_000_000 for ns in legacy],
        "throughput": {"legacy": [100.0, 102.0, 98.0], "runtime": [120.0, 118, 121]},
    }
    bounds = el.intervals(faster, replicates=200)
    assert bounds["p50_ms"] == [-2.0, -2.0] and bounds["p95_ms"] == [-2.0, -2.0]
    assert 0 < bounds["per_s"][0] <= bounds["per_s"][1]


def test_intervals_straddle_zero_when_the_sides_match(el):
    calls = [int((10 + i % 7) * 1e6) for i in range(60)]
    same = el.intervals({"legacy": calls, "runtime": calls[1:] + calls[:1]})
    assert same["p50_ms"][0] <= 0 <= same["p50_ms"][1]
    assert "per_s" not in same
