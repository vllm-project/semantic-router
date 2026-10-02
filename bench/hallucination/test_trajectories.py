import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bench.hallucination.evaluate_trajectories import evaluate_steps
from bench.hallucination.trajectories import (
    FORMAT,
    AssistantStep,
    TrajectoryFormatError,
    load_trajectories,
    step_checks,
    validate_spans,
)

TOKEN_SPANS_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src/semantic-router/pkg/classification/testdata/token_spans_v1_fixtures.json"
)
# The relabeling TestHTTPClassifyHallucination_GoldenFixtures applies before
# running the same cases against the hallucination detector.
GOLDEN_RELABEL = {
    "PERSON": "HALLUCINATED",
    "EMAIL_ADDRESS": "contradiction",
    "PHONE_NUMBER": "unsupported_addition",
    "ADDRESS": "fabricated_reference",
    "URL": "unsupported",
    "CREDIT_CARD": "contradicted",
    "O": "SUPPORTED",
    "HUMAN_NAME": "HUMAN_NAME",
}


def base_document() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "trajectories": [
            {
                "id": "tests",
                "request": "Run the tests.",
                "steps": [
                    {"role": "tool", "name": "run_command", "output": "1 failed"},
                    {
                        "role": "assistant",
                        "text": "All tests pass.",
                        "unsupported_spans": [
                            {
                                "start": 0,
                                "end": 14,
                                "text": "All tests pass",
                                "label": "contradicted",
                            }
                        ],
                    },
                ],
            }
        ],
    }


def write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "trajectories.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def steps(document: dict[str, Any]) -> list[dict[str, Any]]:
    return document["trajectories"][0]["steps"]


def gold_span(document: dict[str, Any]) -> dict[str, Any]:
    return steps(document)[1]["unsupported_spans"][0]


def test_fixture_has_a_correct_final_answer_after_an_unsupported_step() -> None:
    def hides_earlier_claim(assistant: list[AssistantStep]) -> bool:
        return not assistant[-1].unsupported_spans and any(
            step.unsupported_spans for step in assistant[:-1]
        )

    trajectories = load_trajectories()
    assert any(
        hides_earlier_claim(
            [step for step in trajectory.steps if isinstance(step, AssistantStep)]
        )
        for trajectory in trajectories
    )


def test_step_context_excludes_later_tool_output() -> None:
    trajectory = next(
        trajectory
        for trajectory in load_trajectories()
        if trajectory.id == "flaky-test-reported-early"
    )
    early, final = step_checks(trajectory)
    assert "1 failed, 5 passed" in early.context
    assert "6 passed" not in early.context
    assert "6 passed" in final.context
    assert early.question == final.question == trajectory.request


def test_base_document_loads(tmp_path: Path) -> None:
    (trajectory,) = load_trajectories(write(tmp_path, base_document()))
    assert [check.gold_spans for check in step_checks(trajectory)] == [
        (gold_span(base_document()),)
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda d: gold_span(d).update(text="All test pass"),
            "!= message",
            id="text-mismatch",
        ),
        pytest.param(
            lambda d: gold_span(d).update(end=99), "code points", id="end-past-text"
        ),
        pytest.param(
            lambda d: gold_span(d).update(start=14), "code points", id="empty-span"
        ),
        pytest.param(
            lambda d: gold_span(d).update(start=True), "code points", id="bool-offset"
        ),
        pytest.param(
            lambda d: gold_span(d).update(label="SUPPORTED"),
            "marks supported text",
            id="outside-label",
        ),
        pytest.param(
            lambda d: gold_span(d).update(label="PERSON"),
            "unknown label",
            id="unknown-label",
        ),
        pytest.param(
            lambda d: gold_span(d).update(score=1.5), "score", id="score-out-of-range"
        ),
        pytest.param(
            lambda d: steps(d)[1]["unsupported_spans"].append(
                copy.deepcopy(gold_span(d))
            ),
            "duplicate span",
            id="duplicate-span",
        ),
        pytest.param(
            lambda d: steps(d)[1].pop("unsupported_spans"),
            "missing keys",
            id="unlabeled-step",
        ),
        pytest.param(
            lambda d: steps(d)[1].update(unsuported_spans=[]),
            "unknown keys",
            id="misspelled-key",
        ),
        pytest.param(
            lambda d: steps(d).reverse(), "no earlier tool output", id="no-context"
        ),
        pytest.param(
            lambda d: d.update(format="agent_trajectory.v0"), "format", id="format"
        ),
        pytest.param(
            lambda d: d["trajectories"].append(copy.deepcopy(d["trajectories"][0])),
            "duplicate trajectory ids",
            id="duplicate-id",
        ),
    ],
)
def test_loader_rejects_invalid_fixture(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    document = base_document()
    mutate(document)
    with pytest.raises(TrajectoryFormatError, match=message):
        load_trajectories(write(tmp_path, document))


def golden_cases() -> list[dict[str, Any]]:
    return json.loads(TOKEN_SPANS_FIXTURES.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", golden_cases(), ids=lambda case: case["name"])
def test_span_rules_match_token_spans_golden_fixtures(case: dict[str, Any]) -> None:
    spans = [
        {
            **{key: value for key, value in span.items() if not key.startswith("_")},
            "label": GOLDEN_RELABEL[span["label"]],
        }
        for span in case["spans"]
    ]
    if case["expect"] == "reject":
        with pytest.raises(TrajectoryFormatError):
            validate_spans(spans, text=case["text"], where=case["name"])
    else:
        validate_spans(spans, text=case["text"], where=case["name"])


class StubDetector:
    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def predict(
        self, *, context: list[str], question: str, answer: str, output_format: str
    ) -> list[dict[str, Any]]:
        self.calls.append({"context": context, "question": question, "answer": answer})
        return self.responses.get(answer, [])


def test_evaluate_steps_scores_every_assistant_step() -> None:
    trajectories = load_trajectories()
    checks = [check for trajectory in trajectories for check in step_checks(trajectory)]
    unsupported = [check for check in checks if check.gold_spans]
    supported = [check for check in checks if not check.gold_spans]
    caught, false_alarm = unsupported[0], supported[0]
    detector = StubDetector(
        {
            caught.answer: list(caught.gold_spans),
            false_alarm.answer: [
                {"start": 0, "end": 4, "text": false_alarm.answer[:4]}
            ],
        }
    )

    metrics, rows = evaluate_steps(detector=detector, trajectories=trajectories)

    assert [call["context"] for call in detector.calls] == [
        [check.context] for check in checks
    ]
    assert metrics["step_level"]["tp"] == 1
    assert metrics["step_level"]["fn"] == len(unsupported) - 1
    assert metrics["step_level"]["fp"] == 1
    assert metrics["step_level"]["tn"] == len(supported) - 1
    assert [row["outcome"] for row in rows].count("tp") == 1
