import pytest
from vllm_srun.errors import INVALID_QUESTION, MAX_LENGTH_EXCEEDED, QuestionError
from vllm_srun.systemone import (
    canonical,
    json_payload,
    question_options,
    valid_state,
)
from vllm_srun.text.segments import SUFFIX, encode, segments


def test_segments_reproduce_the_scored_prompt():
    prefix, options, suffix = segments(
        {"b": 1, "a": "é"},
        "choice",
        "Pick one",
        [{"key": "x", "description": None}, {"key": "y", "description": [1]}],
    )
    assert (
        prefix
        == 'Context:\n{"a":"é","b":1}\n\nTask type: choice\nQuestion:\nPick one\nOptions:'
    )
    assert options == [
        '\n<option>\n{"description":null,"key":"x"}\n</option>',
        '\n<option>\n{"description":[1],"key":"y"}\n</option>',
    ]
    assert suffix == SUFFIX


def test_noul_defaults_and_partial_criteria():
    kind, _, options = question_options({"type": "noul", "instructions": "q"})
    assert kind == "noul"
    assert options == [
        {"key": "false", "description": "No"},
        {"key": "true", "description": "Yes"},
    ]
    _, _, options = question_options(
        {"type": "noul", "instructions": "q", "criteria": {"true": "Definitely"}}
    )
    assert options == [
        {"key": "false", "description": "No"},
        {"key": "true", "description": "Definitely"},
    ]
    _, _, options = question_options(
        {"type": "noul", "instructions": "q", "criteria": {"false": None}}
    )
    assert options[0] == {"key": "false", "description": "No"}


def test_score_levels_are_ordered_indices():
    _, _, options = question_options(
        {"type": "score", "instructions": "q", "criteria": ["a", {"b": 1}]}
    )
    assert [o["key"] for o in options] == ["0", "1"]
    _, _, same = question_options(
        {"type": "score", "instructions": "q", "levels": ["a", {"b": 1}]}
    )
    assert same == options


def test_choices_superset_matches_criteria_order():
    _, _, from_criteria = question_options(
        {"type": "choice", "instructions": "q", "criteria": {"b": "B", "a": None}}
    )
    _, _, from_choices = question_options(
        {
            "type": "choice",
            "instructions": "q",
            "choices": [{"key": "b", "description": "B"}, {"key": "a"}],
        }
    )
    assert from_choices == from_criteria


@pytest.mark.parametrize(
    "question",
    [
        None,
        {"type": "set", "instructions": "q"},
        {"type": "choice", "instructions": ""},
        {"type": "choice", "instructions": "q", "criteria": {"only": "one"}},
        {
            "type": "choice",
            "instructions": "q",
            "criteria": {f"k{i}": "d" for i in range(256)},
        },
        {
            "type": "choice",
            "instructions": "q",
            "criteria": {"a": float("nan"), "b": "x"},
        },
        {"type": "noul", "instructions": "q", "criteria": {"yes": "y", "no": "n"}},
        {"type": "noul", "instructions": "q", "criteria": {"false": "", "true": "y"}},
        {"type": "choice", "instructions": " ", "criteria": {"a": "A", "b": "B"}},
        {"type": "noul", "instructions": "q", "colour": "blue"},
        {"type": "score", "instructions": "q", "criteria": ["only"]},
        {"type": "score", "instructions": "q", "criteria": [str(i) for i in range(11)]},
        {
            "type": "score",
            "instructions": "q",
            "criteria": ["a", "b"],
            "levels": ["a", "b"],
        },
        {
            "type": "choice",
            "instructions": "q",
            "criteria": {"a": 1, "b": 2},
            "choices": [{"key": "a"}],
        },
        {
            "type": "choice",
            "instructions": "q",
            "choices": [{"key": "a"}, {"key": "a"}],
        },
        {"type": "choice", "instructions": "q", "levels": ["a", "b"]},
    ],
)
def test_invalid_questions(question):
    with pytest.raises(QuestionError) as error:
        question_options(question)
    assert error.value.code == INVALID_QUESTION


def test_overlong_questions_are_rejected_never_truncated():
    def tokens(text):
        return list(range(len(text)))

    with pytest.raises(QuestionError) as error:
        encode(
            "q",
            "x" * 100,
            "noul",
            "q",
            [
                {"key": "false", "description": "No"},
                {"key": "true", "description": "Yes"},
            ],
            tokens,
            64,
        )
    assert error.value.code == MAX_LENGTH_EXCEEDED


def test_endpoints_and_query_positions():
    encoded = encode(
        "q",
        "s",
        "choice",
        "i",
        [{"key": "a", "description": "A"}, {"key": "b", "description": "B"}],
        lambda text: [ord(c) for c in text],
        10_000,
    )
    prefix, options, suffix = segments(
        "s",
        "choice",
        "i",
        [{"key": "a", "description": "A"}, {"key": "b", "description": "B"}],
    )
    assert encoded["endpoints"] == [
        len(prefix) + len(options[0]) - 1,
        len(prefix) + len(options[0]) + len(options[1]) - 1,
    ]
    assert (
        encoded["query"]
        == len(prefix) + len(options[0]) + len(options[1]) + len(suffix) - 1
    )


def test_payload_rules():
    assert (
        json_payload("x") and json_payload({"a": [1, None, True]}) and json_payload([])
    )
    assert not json_payload(None) and json_payload(None, nullable=True)
    assert (
        not json_payload(1)
        and not json_payload({"a": float("inf")})
        and not json_payload({1: "x"})
    )
    assert valid_state("text") and valid_state({"a": 1}) and not valid_state(3)
    assert canonical({"b": 1, "a": [1.5]}) == '{"a":[1.5],"b":1}'
