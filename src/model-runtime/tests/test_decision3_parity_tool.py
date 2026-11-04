"""The parity record must fail when outputs drift or requests were not answered."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "decision3_parity", Path(__file__).parents[1] / "tools/decision3_parity.py"
)
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def record():
    return {
        "id": "case",
        "question_ids": ["q"],
        "body_sha256": "digest",
        "status": 200,
        "stable": True,
        "warm_ms": 1.0,
        "response": {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.75, "b": 0.25},
                }
            },
            "usage": {"input_tokens": 12},
        },
    }


@pytest.mark.parametrize(
    "change", ["probability", "choice", "usage", "status", "unstable", "missing-answer"]
)
def test_actual_drift_fails(change):
    reference, actual = record(), record()
    if change == "probability":
        actual["response"]["answers"]["q"]["probabilities"]["a"] += 1e-7
    elif change == "choice":
        actual["response"]["answers"]["q"]["choice"] = "b"
    elif change == "usage":
        actual["response"]["usage"]["input_tokens"] += 1
    elif change == "status":
        actual["status"] = 503
    elif change == "unstable":
        actual["stable"] = False
    else:
        actual["response"]["answers"] = {}
    assert parity.compare([reference], [actual])["passed"] == 0


def test_success_and_request_identity():
    reference = record()
    assert parity.compare([reference], [copy.deepcopy(reference)])["passed"] == 1
    actual = copy.deepcopy(reference)
    actual["body_sha256"] = "different"
    with pytest.raises(ValueError, match="request differs"):
        parity.compare([reference], [actual])


def test_nonfinite_and_identical_failures_are_not_parity():
    assert parity.differences(float("nan"), float("nan"))
    reference = record()
    reference.update(status=500, response={"error": "failed"})
    assert parity.compare([reference], [reference])["passed"] == 0


def test_matching_question_errors_and_missing_answers_do_not_pass():
    for answers in (
        {"q": {"error": "max_length_exceeded"}},
        {},
        {"another": {"type": "noul", "noul": 1.0}},
    ):
        reference = record()
        reference["response"]["answers"] = answers
        assert parity.compare([reference], [reference])["passed"] == 0
