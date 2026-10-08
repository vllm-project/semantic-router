"""Vela 2.0 request reading: typed parts, over, question validation and presets."""

from __future__ import annotations

import pytest
from vllm_srun.families.vela2.calibration import Calibration
from vllm_srun.families.vela2.request import (
    NOUL_DEFAULT_NO,
    NOUL_DEFAULT_YES,
    QuestionReader,
    read_state,
)
from vllm_srun.testing.vela2 import calibration

CHOICE = {
    "type": "choice",
    "instructions": "Pick one",
    "criteria": {"a": "first", "b": None},
}


def reader(broad: bool = False) -> QuestionReader:
    return QuestionReader(Calibration(calibration(decoder=broad)), broad_head=broad)


def test_text_and_json_text_states() -> None:
    assert read_state("hello").parts[0].role == "user"
    state = read_state(' {"request": "hi", "answer": "yes"}')
    assert state.roles == ("user", "answer")
    with pytest.raises(ValueError):
        read_state("   ")


def test_object_state_maps_keys_to_parts_with_field_ranges() -> None:
    state = read_state(
        {"prompt": "Q?", "source": "doc", "notes": {"n": 1}, "response": "A."}
    )
    assert state.roles == ("user", "context", "answer")
    context = state.text("context")
    assert context == 'source:\ndoc\n\nnotes:\n{"n":1}'
    role, start, end = state.fields["notes"]
    assert (role, context[start:end]) == ("context", '{"n":1}')
    assert state.fields["prompt"] == ("user", 0, 2)


def test_array_state_is_canonical_json_in_the_user_part() -> None:
    assert read_state(["a", {"b": 1}]).text("user") == '["a",{"b":1}]'
    with pytest.raises(ValueError):
        read_state({})


def test_noul_becomes_a_two_option_choice_with_abstain() -> None:
    plan = reader().read("text", {"q": {"type": "noul", "instructions": "Yes?"}})
    question = plan.questions[0]
    assert (question.type, question.abstain) == ("choice", True)
    assert question.options == (("no", NOUL_DEFAULT_NO), ("yes", NOUL_DEFAULT_YES))


def test_choice_score_set_and_span_options() -> None:
    plan = reader().read(
        {"request": "r", "answer": "a"},
        {
            "c": CHOICE,
            "s": {
                "type": "score",
                "instructions": "How?",
                "criteria": ["low", {"x": 1}],
            },
            "t": {"type": "set", "instructions": "Which?", "criteria": {"x": "y"}},
            "p": {"type": "span", "instructions": "Spans?", "criteria": {"L": None}},
        },
    )
    by_id = {q.id: q for q in plan.questions}
    assert by_id["c"].options == (("a", "first"), ("b", ""))
    assert by_id["s"].options == (("low", ""), ('{"x":1}', ""))
    assert by_id["t"].abstain is False
    assert by_id["p"].over == "answer"
    assert by_id["c"].over == ("user", "answer")


def test_over_resolves_keys_parts_and_lists() -> None:
    state = {"request": "r", "source": "s", "tools": "t", "answer": "a"}
    plan = reader().read(
        state,
        {
            "k": {**CHOICE, "over": "tools"},
            "l": {**CHOICE, "over": ["answer", "request"]},
            "p": {
                "type": "span",
                "instructions": "x",
                "criteria": {"L": "l"},
                "over": "tools",
            },
        },
    )
    by_id = {q.id: q for q in plan.questions}
    assert by_id["k"].over == "context"
    assert by_id["l"].over == ("user", "answer")
    assert by_id["p"].span_range is not None


def test_invalid_questions_fail_alone() -> None:
    plan = reader().read(
        "text",
        {
            "ok": CHOICE,
            "type": {"type": "rank", "instructions": "x"},
            "few": {"type": "choice", "instructions": "x", "criteria": {"a": "b"}},
            "blank": {"type": "noul", "instructions": "   "},
            "over": {**CHOICE, "over": "missing"},
            "threshold": {
                "type": "set",
                "instructions": "x",
                "criteria": {"a": "b"},
                "threshold": 2,
            },
            "extra": {**CHOICE, "threshold": 0.5},
            "levels": {"type": "score", "instructions": "x", "criteria": ["a", None]},
            "labels": {"type": "set", "instructions": "x", "criteria": {}},
            "unknown": {
                "type": "span",
                "instructions": "x",
                "criteria": {"a": "b"},
                "colour": 1,
            },
            "head": {
                "type": "span",
                "instructions": "x",
                "criteria": {"a": "b"},
                "head": "broad",
            },
        },
    )
    assert [q.id for q in plan.questions] == ["ok"]
    assert set(plan.errors) == set(plan.question_ids) - {"ok"}
    assert all(error["error"] == "invalid_question" for error in plan.errors.values())


def test_a_span_reads_one_part() -> None:
    span = {"type": "span", "instructions": "x", "criteria": {"a": "b"}}
    plan = reader().read(
        {"request": "r", "answer": "a"},
        {
            "one": {**span, "over": ["answer"]},
            "two": {**span, "over": ["request", "answer"]},
        },
    )
    assert [q.over for q in plan.questions] == ["answer"]
    assert plan.errors["two"]["error"] == "invalid_question"


def test_set_label_views_must_not_collide_with_questions() -> None:
    plan = reader().read(
        "t",
        {
            "s": {"type": "set", "instructions": "x", "criteria": {"a": "b"}},
            "s.a": {"type": "noul", "instructions": "y"},
        },
    )
    assert plan.errors["s"]["error"] == "invalid_question"
    assert [q.id for q in plan.questions] == ["s.a"]


def test_choices_and_levels_superset_fields() -> None:
    plan = reader().read(
        "t",
        {
            "c": {
                "type": "choice",
                "instructions": "x",
                "choices": [{"key": "a"}, {"key": "b", "description": "B"}],
            },
            "s": {"type": "score", "instructions": "x", "levels": ["lo", "hi"]},
        },
    )
    assert [q.options for q in plan.questions] == [
        (("a", ""), ("b", "B")),
        (("lo", ""), ("hi", "")),
    ]


def test_presets_fill_in_the_trained_schemas() -> None:
    plan = reader().read(
        {"request": "q", "source": "d", "answer": "a"},
        {
            "pii": {"preset": "pii", "over": "request", "threshold": 0.3},
            "h": {"preset": "halu"},
            "r": {"preset": "relevance", "over": ["request", "source"]},
        },
    )
    by_id = {q.id: q for q in plan.questions}
    assert by_id["pii"].type == "span" and len(by_id["pii"].options) == 17
    assert by_id["pii"].threshold == 0.3
    assert (by_id["h"].key, by_id["h"].over) == ("halu", "answer")
    assert by_id["r"].options[0][0] == "not_relevant" and by_id["r"].type == "score"
    bad = reader().read(
        "t", {"x": {"preset": "toxic"}, "y": {"preset": "pii", "instructions": "x"}}
    )
    assert set(bad.errors) == {"x", "y"}


def test_span_head_override_needs_a_broad_head() -> None:
    plan = reader(broad=True).read(
        "t",
        {
            "p": {
                "type": "span",
                "instructions": "x",
                "criteria": {"a": "b"},
                "head": "broad",
            }
        },
    )
    assert plan.questions[0].head == "broad"


def test_labels_is_the_router_configs_form_of_set_and_span_criteria() -> None:
    labels = [
        {"key": "billing", "description": "A payment problem"},
        {"key": "shipping"},
    ]
    plan = reader().read(
        "My card was charged twice and the parcel never arrived.",
        {
            "set": {"type": "set", "instructions": "Which apply?", "labels": labels},
            "span": {"type": "span", "instructions": "Find them", "labels": labels},
            "both": {
                "type": "set",
                "instructions": "x",
                "labels": labels,
                "criteria": {"billing": "b"},
            },
            "shape": {"type": "set", "instructions": "x", "labels": {"billing": "b"}},
        },
    )
    by_id = {q.id: q for q in plan.questions}
    assert by_id["set"].options == (("billing", "A payment problem"), ("shipping", ""))
    assert by_id["span"].options == by_id["set"].options
    assert plan.errors["both"]["message"] == "use criteria or labels, not both"
    assert (
        plan.errors["shape"]["message"] == "labels must be a list of {key, description}"
    )


def test_an_invalid_question_says_why() -> None:
    plan = reader().read(
        "text",
        {
            "unknown": {
                "type": "set",
                "instructions": "x",
                "criteria": {"a": "b"},
                "colour": 1,
            },
            "few": {"type": "choice", "instructions": "x", "criteria": {"a": "b"}},
            "s": {"type": "set", "instructions": "x", "criteria": {"a": "b"}},
            "s.a": {"type": "noul", "instructions": "y"},
        },
    )
    assert plan.errors["unknown"] == {
        "type": "set",
        "error": "invalid_question",
        "message": "set questions do not take ['colour']",
    }
    assert plan.errors["few"]["message"] == "criteria must name 2 to 255 options"
    assert "s.<label>" in plan.errors["s"]["message"]
