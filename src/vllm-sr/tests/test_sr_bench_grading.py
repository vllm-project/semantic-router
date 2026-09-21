"""Capability extraction stays distinct from format compliance and old evidence."""

import pytest
from cli.sr_bench.grading import MCQ_GRADER_VERSION, basic_grade
from cli.sr_bench.offline import regrade
from cli.sr_bench.store import Store


@pytest.mark.parametrize("benchmark", ["mmlu-pro", "gpqa-diamond"])
@pytest.mark.parametrize(
    "final, strict",
    [
        ("B", True),
        ("Answer: (B).", True),
        ("**B**", False),
        ("__B__", False),
        ("`B`", False),
        ("**B**\n\nThis follows from the stated conditions.", False),
        ("Reasoning in the visible final. The answer is (B).", False),
        ("The correct answer is **B**.", False),
        ("Final answer: B", False),
        (r"Thus $\boxed{B}$.", False),
        ("B\nThe answer is (B).", False),
    ],
)
def test_explicit_final_answer_and_format_are_independent(benchmark, final, strict):
    grade = basic_grade({"benchmark": benchmark, "answer": "B"}, final)
    assert grade["correct"] is True
    assert grade["answer"] == "B"
    assert grade["details"] == {
        "grader_version": MCQ_GRADER_VERSION,
        "strict_format": strict,
        "answer_status": "parsed",
    }


@pytest.mark.parametrize(
    "final, status",
    [
        ("", "unparsed"),
        ("A is probably the answer", "unparsed"),
        ("A or B", "unparsed"),
        ("Answer: A or B", "ambiguous"),
        ("A and B are both plausible.", "unparsed"),
        ("The answer is banana.", "unparsed"),
        ("Answer: K", "unparsed"),
        ("Analysis mentions B in passing.", "unparsed"),
        ("<think>The answer is B</think>", "unparsed"),
        ("**A**\nFinal answer: B", "ambiguous"),
        ("Answer: A. The answer is B.", "ambiguous"),
        ("Answer: B. Final answer: A, since the premise rules out B.", "ambiguous"),
        ("**B**\nThe answer is A because the conditions hold.", "ambiguous"),
        ("Answer: B. Final answer: A or B.", "ambiguous"),
        ("Answer: A, B", "ambiguous"),
        ("Answer: B and A", "ambiguous"),
        ("A\nB", "ambiguous"),
        (r"\boxed{A} and \boxed{B}", "ambiguous"),
    ],
)
def test_no_guessing_from_prose_or_conflicting_answers(final, status):
    grade = basic_grade({"benchmark": "mmlu-pro", "answer": "B"}, final)
    assert grade["answer"] is None
    assert grade["correct"] is False
    assert grade["details"]["answer_status"] == status


def test_wrong_but_well_formed_answer_remains_wrong():
    grade = basic_grade({"benchmark": "gpqa-diamond", "answer": "B"}, "A")
    assert grade["correct"] is False
    assert grade["details"]["strict_format"] is True


def test_versioned_offline_regrade_preserves_historical_scores(tmp_path):
    store = Store(tmp_path)
    case = {"id": "case", "benchmark": "mmlu-pro", "answer": "B"}
    frozen = {
        "mode": "live",
        "cases": [case],
        "plan_sha256": "historical-plan",
        "targets": [{"id": "model", "kind": "single", "model": "model"}],
        "adapter_versions": {"mmlu-pro": "sr-bench-1.0"},
    }
    run, _ = store.create(frozen)
    identity = run["id"]
    store.result(identity, "case", "model", "completed", {"correct": False})
    call_id = store.start_call(identity, "case", "model", "subject", {})
    store.finish_call(call_id, "completed", {"final": "**B**", "output_complete": True})
    store.status(identity, "completed")
    before = {
        "run": store.get(identity),
        "calls": store.calls(identity),
        "results": store.results(identity),
        "events": store.events(identity),
    }
    result = regrade(store, identity)
    assert result["source_adapter_versions"] == {"mmlu-pro": "sr-bench-1.0"}
    assert result["adapter_versions"] == {"mmlu-pro": MCQ_GRADER_VERSION}
    assert result["results"][0]["correct"] is True
    assert result["results"][0]["details"]["strict_format"] is False
    assert result["changed_count"] == 1
    assert result["model_requests"] == 0
    assert before == {
        "run": store.get(identity),
        "calls": store.calls(identity),
        "results": store.results(identity),
        "events": store.events(identity),
    }
