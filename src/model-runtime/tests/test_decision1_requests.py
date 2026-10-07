"""Decision 1.0 System One rules (the bundled runtime's validation, Noul defaults, presets) and answers."""

from __future__ import annotations

import math

import pytest
from vllm_srun.errors import INVALID_QUESTION, QuestionError
from vllm_srun.families.decision1 import qwen, vela
from vllm_srun.families.decision1.answers import (
    answer,
    choice_confidence,
    score_confidence,
)
from vllm_srun.families.decision1.questions import (
    MAX_QUESTIONS,
    Candidate,
    check_request,
    parse,
)

CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": None}}


@pytest.mark.parametrize(
    "state,questions",
    [
        ("text", {}),
        ("text", {f"q{i}": CHOICE for i in range(MAX_QUESTIONS + 1)}),
        ("text", {" ": CHOICE}),
        ("   ", {"q": CHOICE}),
        (None, {"q": CHOICE}),
        (3, {"q": CHOICE}),
        ({"x": float("nan")}, {"q": CHOICE}),
    ],
)
def test_malformed_requests_are_request_errors(state, questions):
    with pytest.raises(ValueError):
        check_request(state, questions)


def test_valid_states():
    for state in ("text", {"request": "hi", "n": 1}, ["a", {"b": None}]):
        check_request(state, {"q": CHOICE})


@pytest.mark.parametrize(
    "question",
    [
        "not an object",
        {"type": "set", "instructions": "x", "criteria": {"a": "A"}},
        {
            "type": "choice",
            "instructions": "x",
            "criteria": {"a": "A", "b": "B"},
            "over": "request",
        },
        {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        {"type": "choice", "instructions": " \n", "criteria": {"a": "A", "b": "B"}},
        {"type": "choice", "instructions": "x", "criteria": {"a": "A"}},
        {"type": "choice", "instructions": "x", "criteria": {" ": "A", "b": "B"}},
        {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": "B"}},
        {"type": "noul", "instructions": "x", "criteria": {"maybe": "M"}},
        {"type": "score", "instructions": "x", "criteria": ["only one"]},
        {"type": "score", "instructions": "x", "criteria": ["low", None]},
        {"type": "score", "instructions": "x", "criteria": {"0": "low", "1": "high"}},
        {"type": "score", "instructions": "x", "choices": [{"key": "0"}, {"key": "1"}]},
        {"type": "choice", "instructions": "x", "levels": ["a", "b"]},
        {
            "type": "choice",
            "instructions": "x",
            "criteria": {"a": "A", "b": "B"},
            "choices": [],
        },
        {
            "type": "choice",
            "instructions": "x",
            "choices": [{"key": "a"}, {"key": "a"}],
        },
        {"preset": "domain", "type": "choice"},
        {"preset": "unknown"},
    ],
)
def test_invalid_questions(question):
    with pytest.raises(QuestionError) as error:
        parse("q", question, vela.NOUL_DEFAULTS, {"domain": CHOICE})
    assert error.value.code == INVALID_QUESTION


def test_choice_score_and_superset_fields():
    row = parse("q", CHOICE, qwen.NOUL_DEFAULTS)
    assert row.kind == "choice" and row.candidates == (
        Candidate("a", "A"),
        Candidate("b", None),
    )
    ordered = {
        "type": "choice",
        "instructions": {"task": "route"},
        "choices": [{"key": "b", "description": "B"}, {"key": "a"}],
    }
    assert parse("q", ordered, qwen.NOUL_DEFAULTS).keys == ["b", "a"]
    score = {"type": "score", "instructions": "How?", "levels": ["low", {"x": 1}]}
    assert parse("q", score, qwen.NOUL_DEFAULTS).candidates == (
        Candidate("0", "low"),
        Candidate("1", {"x": 1}),
    )


def test_noul_defaults_per_runtime():
    bare = {"type": "noul", "instructions": "Is it?"}
    explicit_null = {
        "type": "noul",
        "instructions": "Is it?",
        "criteria": {"true": None},
    }
    encoder = parse("q", explicit_null, vela.NOUL_DEFAULTS)
    assert [candidate.description for candidate in encoder.candidates] == [
        vela.NOUL_DEFAULTS.false,
        vela.NOUL_DEFAULTS.true,
    ]
    decoder = parse("q", explicit_null, qwen.NOUL_DEFAULTS)
    assert [candidate.description for candidate in decoder.candidates] == [
        qwen.NOUL_DEFAULTS.false,
        None,
    ]
    assert parse("q", bare, qwen.NOUL_DEFAULTS).keys == ["false", "true"]
    given = {
        "type": "noul",
        "instructions": "Is it?",
        "criteria": {"true": "Y", "false": "N"},
    }
    assert [
        c.description for c in parse("q", given, vela.NOUL_DEFAULTS).candidates
    ] == ["N", "Y"]


def test_presets_fill_in_the_question():
    presets = {"domain": CHOICE}
    assert parse("q", {"preset": "domain"}, vela.NOUL_DEFAULTS, presets) == parse(
        "q", CHOICE, vela.NOUL_DEFAULTS
    )


def test_choice_answer_uses_the_top_two_margin():
    result = answer("choice", ["a", "b", "c"], ["A", "B", None], [0.5, 0.3, 0.2])
    assert result == {
        "type": "choice",
        "choice": "a",
        "confidence": pytest.approx(0.2),
        "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2},
    }
    tie = answer("choice", ["a", "b"], [None, None], [0.5, 0.5])
    assert tie["choice"] == "a" and tie["confidence"] == 0.0


def test_score_answer_reports_concentration_and_a_text_legend():
    result = answer(
        "score", ["0", "1", "2"], ["low", "mid", {"x": [1]}], [0.0, 0.0, 1.0]
    )
    assert result["score"] == 2.0
    assert result["confidence"] == pytest.approx(1.0 - (0.0 / (8 / 12)))
    assert result["legend"] == {"0": "low", "1": "mid", "2": '{"x":[1]}'}
    uniform = [1 / 3] * 3
    assert score_confidence(uniform) == pytest.approx(0.0, abs=1e-12)
    assert choice_confidence([1.0]) == 1.0


def test_noul_answer_is_p_true():
    assert answer("noul", ["false", "true"], ["N", "Y"], [0.25, 0.75]) == {
        "type": "noul",
        "noul": 0.75,
    }


@pytest.mark.parametrize(
    "values",
    [None, [0.5], [0.6, 0.6], [math.nan, 1.0], [1.5, -0.5], [0.5, 0.49]],
)
def test_malformed_model_output(values):
    assert answer("noul", ["false", "true"], ["N", "Y"], values) == {
        "type": "noul",
        "error": "invalid_model_output",
    }
