"""C6: the mock backend's answer-plan fixture pins model x category outcomes.

The V1 cold-start golden scenario needs replay-tier routing and direct-model
evals to observe the same per-question correctness on the mock stack: the
plan drives "Answer: <letter>." completions with a deterministic wrong
letter when a model is planned to fail a category.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SMOKE_DIR = Path(__file__).resolve().parents[1] / "examples" / "smoke"


@pytest.fixture(scope="module")
def mock():
    spec_file = importlib.util.spec_from_file_location(
        "mock_backend_under_test", _SMOKE_DIR / "mock_backend.py"
    )
    assert spec_file is not None and spec_file.loader is not None
    module = importlib.util.module_from_spec(spec_file)
    spec_file.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_state(mock):
    mock._ANSWER_PLAN.clear()
    yield
    mock._ANSWER_PLAN.clear()


def test_wrong_letter_is_deterministic_and_never_correct(mock):
    assert mock._wrong_letter("B") == "C"
    assert mock._wrong_letter("b") == "C"
    assert mock._wrong_letter("J") == "A"  # cycles past the end
    assert mock._wrong_letter("Z") == "A"  # unknown letters still map somewhere


def test_plan_answer_returns_planned_letters(mock):
    mock._ANSWER_PLAN.update({"local/private-qwen": {"math": True, "history": False}})
    good = {"expected": "B", "category": "math"}
    bad = {"expected": "B", "category": "history"}
    assert mock._plan_answer("local/private-qwen", good) == "Answer: B."
    assert mock._plan_answer("local/private-qwen", bad) == "Answer: C."


def test_plan_answer_star_fallback_and_misses(mock):
    mock._ANSWER_PLAN.update({"m-a": {"*": True}})
    assert (
        mock._plan_answer("m-a", {"expected": "A", "category": "anything"})
        == "Answer: A."
    )
    # explicit category beats the star: unlisted categories fall through
    mock._ANSWER_PLAN.clear()
    mock._ANSWER_PLAN.update({"m-a": {"math": True}})
    assert mock._plan_answer("m-a", {"expected": "A", "category": "law"}) is None
    # unplanned model, unplanned category-less case, or case without an
    # expected letter all fall through to the legacy behavior
    assert mock._plan_answer("m-b", {"expected": "A", "category": "math"}) is None
    assert mock._plan_answer("m-a", {"expected": "A"}) is None
    assert mock._plan_answer("m-a", {"category": "math"}) is None


def test_answer_for_plan_beats_capability_table(mock):
    # planned completions take precedence over the truth-based fallback
    mock._CAPABILITY_TABLE.clear()
    case = {
        "prompt": "2+2?",
        "expected": "B",
        "category": "math",
        "truth": "cloud_frontier_reasoning",  # legacy rule: local lane fails
        "expected_contains": "4",
    }
    mock._CAPABILITY_TABLE[case["prompt"]] = case
    mock._ANSWER_PLAN.update({"local/private-qwen": {"math": True}})
    assert mock._answer_for("local/private-qwen", "2+2?") == "Answer: B."
    # an unplanned model keeps the legacy capability behavior
    assert "deeper analysis" in mock._answer_for("local/private-qwen-x", "2+2?")


def test_answer_for_unplanned_behavior_unchanged(mock):
    mock._CAPABILITY_TABLE.clear()
    mock._ANSWER_PLAN.update({"local/private-qwen": {"math": True}})
    # no capability entry: the fixed ANSWER_TABLE path still applies
    assert "Paris" in mock._answer_for(
        "local/private-qwen", "What is the capital of france?"
    )
