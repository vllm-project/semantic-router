"""The Decision Index engine's answer checks (``tools/index_engine.py``)."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "index_engine.py"
MODEL = "vllm-sr/Decision-2.0-Kai-0.6B"
QUESTIONS = {
    "pick": {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": "A", "b": "B"},
    },
    "flag": {"type": "noul", "instructions": "Is it?"},
}


@pytest.fixture(scope="module")
def engine():
    spec = importlib.util.spec_from_file_location("index_engine", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(**answers):
    body = {
        "pick": {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.7, "b": 0.3},
        },
        "flag": {"type": "noul", "noul": 0.2},
    }
    body.update(answers)
    return {"model": MODEL, "answers": body}


def test_valid_answers_pass_unchanged(engine):
    valid = response()
    assert engine.check(valid, QUESTIONS, MODEL) is valid


def test_a_refused_question_refuses_the_row(engine):
    refused = response(flag={"type": "noul", "error": "max_length_exceeded"})
    with pytest.raises(engine.Refused, match="max_length_exceeded"):
        engine.check(refused, QUESTIONS, MODEL)


@pytest.mark.parametrize(
    "answers",
    [
        {"flag": {"type": "noul", "error": "invalid_model_output"}},
        {"flag": {"type": "noul", "noul": 1.5}},
        {"pick": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7}}},
        {
            "pick": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.7, "b": 0.2},
            }
        },
        {
            "pick": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"a": 0.7, "b": 0.3},
            }
        },
        {"pick": {"type": "noul", "noul": 0.5}},
    ],
)
def test_a_malformed_answer_is_an_error(engine, answers):
    with pytest.raises(ValueError) as raised:
        engine.check(response(**answers), QUESTIONS, MODEL)
    assert not isinstance(raised.value, engine.Refused)


def test_missing_answers_or_another_model_are_errors(engine):
    partial = response()
    del partial["answers"]["flag"]
    with pytest.raises(ValueError, match="missing or extra"):
        engine.check(partial, QUESTIONS, MODEL)
    with pytest.raises(ValueError, match="another model"):
        engine.check(response(), QUESTIONS, "vllm-sr/Decision-2.0-Eos-0.8B")


def test_a_tie_goes_to_the_first_option_in_the_callers_order(engine):
    tie = {"a": 0.5, "b": 0.5}
    first = response(pick={"type": "choice", "choice": "a", "probabilities": tie})
    assert engine.check(first, QUESTIONS, MODEL) is first
    second = response(pick={"type": "choice", "choice": "b", "probabilities": tie})
    with pytest.raises(ValueError, match="first most probable"):
        engine.check(second, QUESTIONS, MODEL)
