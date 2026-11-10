"""The questions the direct Router signal tool sends a model (``tools/router_signal_direct.py``)."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "router_signal_direct.py"

TRUNCATING = {
    "domain_classifier:domain",
    "fact_check_classifier:factcheck",
    "modality_detector:modality",
    "feedback_detector:feedback",
}

VELA2_CARD_LIMITS = {
    "max_input_tokens": 8192,
    "max_scan_tokens": 65536,
    "truncate_tokens": 2048,
    "min_options": 2,
    "max_options": 32,
}
D3_CARD_LIMITS = {
    "max_input_tokens": 32768,
    "min_options": 2,
    "max_options": 32,
    "min_levels": 2,
    "max_levels": 10,
}


@pytest.fixture(scope="module")
def direct():
    spec = importlib.util.spec_from_file_location("router_signal_direct", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_a_card_with_a_scan_budget_truncates(direct) -> None:
    assert direct.has_scan_budget(VELA2_CARD_LIMITS) is True
    assert direct.has_scan_budget(D3_CARD_LIMITS) is False


@pytest.mark.parametrize("own", [False, True])
def test_truncating_marks_exactly_the_questions_not_read_whole(direct, own) -> None:
    for task in direct.TASKS:
        questions = direct.bundle(task, own=own, truncate=True)
        truncating = {
            q for q, body in questions.items() if body.get("overflow") == "truncate"
        }
        assert truncating == TRUNCATING & questions.keys(), (task, own)


@pytest.mark.parametrize("own", [False, True])
def test_without_truncating_the_same_questions_carry_no_overflow(direct, own) -> None:
    for task in direct.TASKS:
        truncating = direct.bundle(task, own=own, truncate=True)
        assert direct.bundle(task, own=own, truncate=False) == {
            q: {k: v for k, v in body.items() if k != "overflow"}
            for q, body in truncating.items()
        }, (task, own)
