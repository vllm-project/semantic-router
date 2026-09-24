"""Golden input-rendering tests for owned Decision model backends."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.contracts import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)
from decision_runtime.model_inputs import (  # noqa: E402
    QWEN_DEFAULT_NO,
    QWEN_DEFAULT_YES,
    VELA_DEFAULT_NO,
    VELA_DEFAULT_YES,
    build_model_input,
    qwen_segments,
    vela_text_input,
)

QWEN_POLICY = {
    "choice_null_description": "preserve_json_null",
    "noul_default_false": QWEN_DEFAULT_NO,
    "noul_default_true": QWEN_DEFAULT_YES,
    "noul_explicit_null": "preserve_json_null",
}
VELA_POLICY = {
    "choice_null_description": "render_key",
    "noul_default_false": VELA_DEFAULT_NO,
    "noul_default_true": VELA_DEFAULT_YES,
    "noul_explicit_null": "use_default",
}


def test_qwen_pointer_v2_segments_preserve_exact_boundaries_and_json() -> None:
    question = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": {"goal": "存活", "weight": 2},
            "criteria": {"left": None, "right": {"lines": 1}},
        }
    )
    row = build_model_input(
        question_id="move-id-must-not-render",
        state={"board": ["..", "##"]},
        question=question,
        **QWEN_POLICY,
    )

    segments = qwen_segments(row)

    assert segments.prefix == (
        'Context:\n{"board":["..","##"]}\n\n'
        'Task type: choice\nQuestion:\n{"goal":"存活","weight":2}\nOptions:'
    )
    assert segments.options == (
        '\n<option>\n{"description":null,"key":"left"}\n</option>',
        '\n<option>\n{"description":{"lines":1},"key":"right"}\n</option>',
    )
    assert segments.suffix.endswith("\nDecision:")
    assert "move-id-must-not-render" not in segments.rendered


def test_nox_choice_null_policy_renders_key_without_mutating_order() -> None:
    question = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"left": None, "right": "Right lane"},
        }
    )

    row = build_model_input(
        question_id="move",
        state="board",
        question=question,
        **{**QWEN_POLICY, "choice_null_description": "render_key"},
    )

    assert [(item.key, item.description) for item in row.candidates] == [
        ("left", None),
        ("right", "Right lane"),
    ]
    assert '"description":"left"' in qwen_segments(row).options[0]


def test_vela_choice_and_score_match_released_marker_text() -> None:
    choice = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": "Choose a lane.",
            "criteria": {
                "left": None,
                "right": "Clear one line",
                "same": "same",
            },
        }
    )
    choice_row = build_model_input(
        question_id="opaque",
        state={"height": 3, "holes": 0},
        question=choice,
        **VELA_POLICY,
    )
    assert vela_text_input(choice_row).candidates == (
        "left",
        "right: Clear one line",
        "same: same",
    )
    assert vela_text_input(choice_row).state == '{"height":3,"holes":0}'

    score = ScoreQuestion.model_validate(
        {
            "type": "score",
            "instructions": "Rate risk.",
            "criteria": ["low", {"label": "high"}],
        }
    )
    score_row = build_model_input(
        question_id="risk",
        state="state",
        question=score,
        **VELA_POLICY,
    )
    assert vela_text_input(score_row).candidates == (
        "level 0: low",
        'level 1: {"label":"high"}',
    )


def test_noul_defaults_and_custom_criteria_preserve_false_true_order() -> None:
    default = NoulQuestion.model_validate({"type": "noul", "instructions": "Escalate?"})
    row = build_model_input(
        question_id="urgent",
        state="state",
        question=default,
        **VELA_POLICY,
    )
    assert [(item.key, item.description) for item in row.candidates] == [
        ("false", VELA_DEFAULT_NO),
        ("true", VELA_DEFAULT_YES),
    ]

    custom = NoulQuestion.model_validate(
        {
            "type": "noul",
            "instructions": "Escalate?",
            "criteria": {"false": "Keep automated", "true": {"route": "human"}},
        }
    )
    custom_row = build_model_input(
        question_id="urgent",
        state="state",
        question=custom,
        **VELA_POLICY,
    )
    assert vela_text_input(custom_row).candidates == (
        "Keep automated",
        '{"route":"human"}',
    )


def test_vela_fails_closed_if_a_profile_allows_null_candidate_text() -> None:
    question = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"left": None, "right": None},
        }
    )
    row = build_model_input(
        question_id="move",
        state="state",
        question=question,
        **QWEN_POLICY,
    )

    with pytest.raises(ValueError, match="must not be null"):
        vela_text_input(row)


def test_noul_defaults_and_explicit_null_follow_model_policy() -> None:
    default = NoulQuestion.model_validate({"type": "noul", "instructions": "Go?"})
    qwen_default = build_model_input(
        question_id="go", state="state", question=default, **QWEN_POLICY
    )
    assert [(item.key, item.description) for item in qwen_default.candidates] == [
        ("false", QWEN_DEFAULT_NO),
        ("true", QWEN_DEFAULT_YES),
    ]

    explicit_null = NoulQuestion.model_validate(
        {
            "type": "noul",
            "instructions": "Go?",
            "criteria": {"false": None, "true": None},
        }
    )
    qwen_null = build_model_input(
        question_id="go", state="state", question=explicit_null, **QWEN_POLICY
    )
    assert [item.description for item in qwen_null.candidates] == [None, None]
    assert '"description":null' in qwen_segments(qwen_null).options[0]

    vela_null = build_model_input(
        question_id="go", state="state", question=explicit_null, **VELA_POLICY
    )
    assert [item.description for item in vela_null.candidates] == [
        VELA_DEFAULT_NO,
        VELA_DEFAULT_YES,
    ]


def test_model_input_deeply_snapshots_mutable_request_content() -> None:
    state = {"board": [".."]}
    question = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": {"goal": ["survive"]},
            "criteria": {"left": {"lines": [1]}, "right": None},
        }
    )
    row = build_model_input(
        question_id="move", state=state, question=question, **VELA_POLICY
    )
    before = vela_text_input(row)

    state["board"].append("##")
    question.instructions["goal"].append("mutated")
    question.criteria["left"]["lines"].append(2)

    assert vela_text_input(row) == before
