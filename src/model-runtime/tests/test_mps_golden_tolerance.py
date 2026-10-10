"""MPS qualification must never borrow an unrecorded generic GPU tolerance."""

import pytest
from vllm_srun.supervision.readiness import compare_numbers, golden_check


def run(_surface, _body):
    return {"score": 0.501}


def compare(_surface, values, reference, tolerance):
    return compare_numbers(values, reference, tolerance)


def golden(tolerance):
    return {
        "surface": "classify",
        "body": {},
        "expected": {"mps": {"score": 0.5}},
        "tolerances": {"mps": tolerance},
    }


@pytest.mark.parametrize("tolerance", [None, -1, float("nan"), float("inf"), True])
def test_mps_reference_needs_a_valid_recorded_tolerance(tolerance):
    result = golden_check(
        run,
        compare,
        [golden(tolerance)],
        "mps",
    )
    assert result.status == "failed"
    assert "recorded per-model tolerance" in result.detail


def test_mps_uses_the_model_tolerance_instead_of_the_gpu_default():
    assert golden_check(run, compare, [golden(0.002)], "mps").status == "matched"
    assert golden_check(run, compare, [golden(0.0001)], "mps").status == "failed"


def test_missing_mps_answers_are_not_reported_as_qualified():
    result = golden_check(
        run, compare, [{"surface": "classify", "body": {}, "expected": {}}], "mps"
    )
    assert result.status == "unverified"
    assert result.reference is None
