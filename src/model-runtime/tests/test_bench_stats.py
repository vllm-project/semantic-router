from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_stats import (
    inconclusive_worse,
    minus_baseline,
    should_extend,
    t_interval,
    verdict,
)


def test_one_round_is_insufficient_without_an_interval():
    interval = t_interval([2.0])
    assert interval["diff"] == 2.0
    assert interval["low"] is None
    assert interval["high"] is None
    assert verdict(interval) == "Insufficient"
    assert inconclusive_worse(interval) is False


def test_verdict_level_when_interval_straddles_zero():
    interval = t_interval([1.0, -0.5, 0.2, 0.4, -0.1])
    assert verdict(interval) == "Level"
    assert inconclusive_worse(interval) is (interval["diff"] > 0)


def test_four_rounds_have_an_interval_but_not_an_acceptance_verdict():
    interval = t_interval([-1.0, -1.1, -0.9, -1.2])
    assert interval["low"] is not None
    assert verdict(interval) == "Insufficient"


def test_minus_baseline_aligns_rounds():
    side = [{"p50_ms": 21.0}, {"p50_ms": 22.0}]
    base = [{"p50_ms": 20.0}, {"p50_ms": 20.0}]
    out = minus_baseline(side, base, ("p50_ms",))
    assert out["rounds"] == 2
    assert out["p50_ms"]["diff"] == 1.5
    assert out["p50_ms"]["verdict"] == "Insufficient"


def test_should_extend_when_short_of_five_rounds():
    out = minus_baseline(
        [{"p50_ms": 21.0}],
        [{"p50_ms": 20.0}],
        ("p50_ms",),
    )
    assert should_extend(out, ("p50_ms",)) is True


@pytest.mark.parametrize("side,base", [([], []), ([{"p50_ms": 1.0}], [])])
def test_minus_baseline_rejects_missing_rounds(side, base):
    with pytest.raises(ValueError):
        minus_baseline(side, base, ("p50_ms",))


def test_empty_interval_is_not_evidence():
    with pytest.raises(ValueError, match="at least one"):
        t_interval([])


@pytest.mark.parametrize("direction,expected", [(-1, True), (1, False)])
def test_throughput_repeat_policy_uses_higher_is_better(direction, expected):
    differences = [direction * value for value in [1.0, -0.5, 0.2, 0.4, -0.1]]
    result = minus_baseline(
        [{"requests_per_second": 20.0 + value} for value in differences],
        [{"requests_per_second": 20.0} for _ in differences],
        ("requests_per_second",),
    )
    assert result["requests_per_second"]["verdict"] == "Level"
    assert should_extend(result, ("requests_per_second",)) is expected
