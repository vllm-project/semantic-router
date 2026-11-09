"""NLI answers prove full pair coverage only for successful strict questions."""

import pytest
from vllm_srun.plugins.base import DEADLINE, SurfaceRequest

from . import test_nli
from .test_api_contract import check
from .test_nli import QUESTIONS, STATE, decisions

# Register the same tiny-package fixtures when this module runs on its own.
packages = test_nli.packages
runtime = test_nli.runtime
client = test_nli.client


def strict_questions():
    return {
        key: {**question, "require_full_input": True}
        for key, question in QUESTIONS.items()
    }


@pytest.mark.parametrize("path", ["/v1/decisions", "/v1/systemone"])
def test_strict_answers_preserve_scores_and_cache_isolation(client, path):
    ordinary = decisions(client, path=path)["answers"]
    for _ in range(2):
        strict = decisions(client, questions=strict_questions(), path=path)["answers"]
        for key, answer in strict.items():
            assert answer.pop("input_coverage") == "complete"
            assert answer == ordinary[key]
        assert decisions(client, path=path)["answers"] == ordinary


def test_mixed_questions_and_composed_noul_items_preserve_requirements(client):
    questions = {
        **QUESTIONS,
        **{
            f"categories:label:{index}": {
                "type": "noul",
                "instructions": f"This request is about {label}.",
                "require_full_input": True,
            }
            for index, label in enumerate(("programming", "travel planning"))
        },
        "strict-choice": strict_questions()["domain"],
        "explicit-false": {**QUESTIONS["code"], "require_full_input": False},
    }
    answers = decisions(client, questions=questions)["answers"]
    for key, question in questions.items():
        assert "error" not in answers[key]
        if question.get("require_full_input"):
            assert answers[key]["input_coverage"] == "complete"
        else:
            assert "input_coverage" not in answers[key]


@pytest.mark.parametrize("state", [{"request": STATE}, [STATE, "More details."]])
def test_structured_state_has_complete_coverage(client, state):
    answers = decisions(client, state=state, questions=strict_questions())["answers"]
    assert all(answer["input_coverage"] == "complete" for answer in answers.values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"require_full_input": "true"},
        {"require_full_input": 1},
        {"require_full_input": None},
        {"require_full_input": True, "overflow": "truncate"},
    ],
)
def test_invalid_strict_requirement_never_gets_proof(client, overrides):
    answers = decisions(
        client,
        questions={**QUESTIONS, "invalid": {**QUESTIONS["code"], **overrides}},
    )["answers"]
    assert answers["invalid"]["error"] == "invalid_question"
    assert "input_coverage" not in answers["invalid"]


def test_one_overlong_candidate_invalidates_the_whole_choice(client, runtime):
    model = next(s.model for s in runtime.served if s.config.name == "nli")
    questions = strict_questions()
    questions["domain"] = {
        **questions["domain"],
        "criteria": {"short": "programming", "long": "travel " * model.limit},
    }
    answers = decisions(client, questions=questions)["answers"]
    assert answers["domain"]["error"] == "max_length_exceeded"
    assert "input_coverage" not in answers["domain"]
    assert answers["code"]["input_coverage"] == "complete"


def test_bundle_and_multi_state_do_not_share_coverage_proofs(client):
    response = client.post(
        "/v1/bundle",
        json={
            "tasks": [
                {
                    "id": "strict",
                    "decisions": {
                        "model": "nli",
                        "state": STATE,
                        "questions": strict_questions(),
                        "states": {
                            "ordinary": {"state": STATE, "questions": QUESTIONS},
                            "long": {
                                "state": STATE * 2000,
                                "questions": strict_questions(),
                            },
                        },
                    },
                },
                {
                    "id": "ordinary",
                    "decisions": {
                        "model": "nli",
                        "state": STATE,
                        "questions": QUESTIONS,
                    },
                },
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    check("BundleResponse", body)
    results = {item["id"]: item["decisions"] for item in body["results"]}
    strict = results["strict"]
    assert all(a["input_coverage"] == "complete" for a in strict["answers"].values())
    ordinary_answers = [
        *strict["states"]["ordinary"]["answers"].values(),
        *results["ordinary"]["answers"].values(),
    ]
    assert all("input_coverage" not in a and "error" not in a for a in ordinary_answers)
    for answer in strict["states"]["long"]["answers"].values():
        assert answer["error"] == "max_length_exceeded"
        assert "input_coverage" not in answer


@pytest.mark.parametrize("failure", ["deadline", "partial-deadline", "bad-logits"])
def test_failed_readouts_never_get_full_input_proof(runtime, failure):
    model = next(s.model for s in runtime.served if s.config.name == "nli")
    plan = model.plan_surface(
        "decisions",
        SurfaceRequest(
            "decisions",
            {"state": STATE, "questions": strict_questions()},
            None,
            "exact",
            False,
            0,
        ),
    )
    results = [(0.0, 1.0)] * len(plan.items)
    if failure == "deadline":
        answers = model.finish_surface(plan, DEADLINE)["answers"]
    else:
        results[1] = DEADLINE if failure == "partial-deadline" else (float("nan"), 1.0)
        answers = model.finish_surface(plan, results)["answers"]
        assert answers["code"]["input_coverage"] == "complete"
    assert "error" in answers["domain"]
    assert "input_coverage" not in answers["domain"]
    if failure == "deadline":
        assert answers["code"]["error"] == "deadline_exceeded"
        assert "input_coverage" not in answers["code"]
