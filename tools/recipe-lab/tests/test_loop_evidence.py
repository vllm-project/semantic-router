"""The shared evidence primitives: bounded numbers and token-budget status."""

from __future__ import annotations

import pytest
from recipe_lab.loop_evidence import fit_status, is_number


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, True),
        (3, True),
        (0.25, True),
        (True, False),  # bool is not a numeric reading
        ("0.5", False),
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_is_number_bounds_and_types(value, expected):
    assert is_number(value) is expected


def test_fit_status_output_limit_wins_over_context():
    assert (
        fit_status(12000, 100, 32768, 4096) == "requested_output_exceeds_output_limit"
    )


def test_fit_status_requested_output_exceeds_context():
    assert fit_status(3000, 100, 2048, 4096) == "requested_output_exceeds_context"


def test_fit_status_unknown_inputs_stay_unknown():
    for args in (
        (None, 100, 8192, 4096),
        (256, None, 8192, 4096),
        (256, 100, None, 4096),
        (256, 100, 8192, None),
    ):
        assert fit_status(*args) == "unknown"


def test_fit_status_source_estimate_crossing_and_fitting():
    assert fit_status(256, 8000, 8192, 4096) == "source_token_estimate_exceeds_context"
    assert fit_status(256, 500, 8192, 4096) == "source_token_estimate_within_limits"
