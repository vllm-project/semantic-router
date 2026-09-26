import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bench.hallucination.evaluate_trajectories import (
    ContextWindow,
    evaluate_steps,
    first_false_steps,
)
from bench.hallucination.trajectories import (
    FORMAT,
    AssistantStep,
    ToolStep,
    Trajectory,
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


def test_claim_before_any_tool_output_is_checked_against_no_context(
    tmp_path: Path,
) -> None:
    document = base_document()
    steps(document).insert(
        0,
        {
            "role": "assistant",
            "text": "The tests pass.",
            "unsupported_spans": [
                {
                    "start": 0,
                    "end": 14,
                    "text": "The tests pass",
                    "label": "unsupported",
                }
            ],
        },
    )
    (trajectory,) = load_trajectories(write(tmp_path, document))
    early, late = step_checks(trajectory)
    assert (early.step, early.context, len(early.gold_spans)) == (0, "", 1)
    assert late.context == "1 failed"


def test_fixture_has_an_unsupported_claim_before_any_tool_output() -> None:
    assert any(
        check.gold_spans and not check.context
        for trajectory in load_trajectories()
        for check in step_checks(trajectory)
    )


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


def claims(id_: str, labels: str, output: str = "3 passed") -> Trajectory:
    """One tool step, then an assistant claim per label: W unsupported, R supported."""
    trajectory_steps: list[ToolStep | AssistantStep] = [
        ToolStep(name="run_command", output=output)
    ]
    for index, label in enumerate(labels):
        text = f"{id_} claim {index}"
        span = {"start": 0, "end": len(text), "text": text, "label": "unsupported"}
        trajectory_steps.append(
            AssistantStep(text=text, unsupported_spans=(span,) if label == "W" else ())
        )
    return Trajectory(id=id_, request="Run the tests.", steps=tuple(trajectory_steps))


def flagging(*answers: str) -> StubDetector:
    return StubDetector(
        {answer: [{"start": 0, "end": 1, "text": answer[0]}] for answer in answers}
    )


@pytest.mark.parametrize(
    ("labels", "flagged", "expected"),
    [
        pytest.param("WR", [0], (1, 1, 0, False), id="caught-at-first"),
        pytest.param("WWR", [1], (1, 2, 1, False), id="caught-late"),
        pytest.param("WR", [1], (1, None, None, False), id="supported-flag-no-catch"),
        pytest.param("RW", [0, 1], (2, 2, 0, True), id="false-alarm-before"),
    ],
)
def test_first_false_step(
    labels: str,
    flagged: list[int],
    expected: tuple[int, int | None, int | None, bool],
) -> None:
    detector = flagging(*(f"t claim {index}" for index in flagged))
    _, rows = evaluate_steps(detector=detector, trajectories=[claims("t", labels)])
    first_false, first_caught, steps_late, false_alarm = expected
    assert first_false_steps(rows) == [
        {
            "trajectory": "t",
            "first_false_step": first_false,
            "first_caught_step": first_caught,
            "steps_late": steps_late,
            "false_alarm_before": false_alarm,
        }
    ]


def test_first_false_step_summary_skips_supported_trajectories() -> None:
    trajectories = [
        claims("first", "WR"),
        claims("late", "WWR"),
        claims("missed", "WR"),
        claims("clean", "RR"),
    ]
    detector = flagging("first claim 0", "late claim 1", "missed claim 1")

    metrics, rows = evaluate_steps(detector=detector, trajectories=trajectories)

    assert [result["trajectory"] for result in first_false_steps(rows)] == [
        "first",
        "late",
        "missed",
    ]
    assert metrics["first_false_step"] == {
        "trajectories": 3,
        "caught_at_first": 1,
        "caught_late": 1,
        "missed": 1,
        "mean_steps_late": 0.5,
        "false_alarm_before": 0,
    }


@pytest.mark.parametrize(
    ("window", "seen_words"),
    [
        pytest.param(None, 40, id="no-window"),
        # The whitespace counter spends 8 of the 20 tokens on the question and answer.
        pytest.param(20, 12, id="window-keeps-the-start"),
    ],
)
def test_context_window_records_how_much_context_each_step_saw(
    window: int | None, seen_words: int
) -> None:
    words = [f"line{index}" for index in range(40)]
    context = " ".join(words)
    detector = StubDetector({})
    context_window = (
        None
        if window is None
        else ContextWindow(tokens=window, count_tokens=lambda text: len(text.split()))
    )

    metrics, (row,) = evaluate_steps(
        detector=detector,
        trajectories=[claims("long", "R", output=context)],
        context_window=context_window,
    )

    (seen,) = detector.calls[0]["context"]
    truncated = seen_words < len(words)
    assert seen.split() == words[:seen_words]
    assert context.startswith(seen)
    assert (row["context_chars"], row["context_chars_seen"], row["truncated"]) == (
        len(context),
        len(seen),
        truncated,
    )
    assert metrics["context_window_tokens"] == window
    assert metrics["truncated_steps"] == {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": int(truncated),
    }
