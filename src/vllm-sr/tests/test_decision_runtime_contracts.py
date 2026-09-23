"""Strict SystemOne request, response, and confidence contracts."""

import json
import math
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.confidence import normalized_top_confidence  # noqa: E402
from decision_runtime.contracts import (  # noqa: E402
    MAX_BATCH_DECISIONS,
    ChoiceAnswer,
    NoulAnswer,
    ResponseContractError,
    ScoreAnswer,
    SystemOneBatchRequest,
    SystemOneBatchResponse,
    SystemOneBatchResult,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    validate_batch_response_for_request,
    validate_response_for_request,
)


def request_payload():
    return {
        "model": "decision-test",
        "state": {"message": "I was charged twice."},
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this a billing issue?",
            },
            "category": {
                "type": "choice",
                "instructions": "Classify the request.",
                "criteria": {"billing": None, "technical": "A product defect"},
            },
            "urgency": {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": ["Can wait", "Needs attention", "Immediate"],
            },
        },
    }


def response_for(request):
    return SystemOneResponse(
        model="decision-test",
        answers={
            "billing": NoulAnswer(type="noul", noul=0.9),
            "category": ChoiceAnswer(
                type="choice",
                choice="billing",
                confidence=0.6,
                probabilities={"billing": 0.8, "technical": 0.2},
            ),
            "urgency": ScoreAnswer(
                type="score",
                score=1.7,
                confidence=0.7,
                legend={"0": "Can wait", "1": "Needs attention", "2": "Immediate"},
                probabilities={"0": 0.1, "1": 0.1, "2": 0.8},
            ),
        },
        usage=Usage(input_tokens=42, output_tokens=3),
    )


def batch_request_payload():
    request = request_payload()
    return {
        "model": request["model"],
        "states": [
            {"id": "first", "state": request["state"]},
            {"id": "second", "state": "A different customer request."},
        ],
        "questions": request["questions"],
    }


def batch_response_for(request):
    single = response_for(request)
    results = [
        SystemOneBatchResult(
            id=state.id,
            answers=single.answers,
            usage=Usage(input_tokens=index + 1, output_tokens=3),
        )
        for index, state in enumerate(request.states)
    ]
    return SystemOneBatchResponse(
        model=request.model,
        results=results,
        usage=Usage(
            input_tokens=sum(result.usage.input_tokens for result in results),
            output_tokens=sum(result.usage.output_tokens for result in results),
        ),
    )


@pytest.mark.parametrize("state", ["message", {"message": "hi"}, ["hi"]])
def test_state_accepts_only_documented_top_level_content(state):
    payload = request_payload()
    payload["state"] = state
    assert SystemOneRequest.model_validate(payload).state == state


@pytest.mark.parametrize("state", [None, True, 7, 1.5])
def test_state_rejects_non_content_scalars(state):
    payload = request_payload()
    payload["state"] = state
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_model_is_explicit_and_request_fields_are_closed():
    payload = request_payload()
    payload.pop("model")
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("model", ["", " ", "\t\n"])
def test_model_rejects_empty_or_whitespace_ids(model):
    payload = request_payload()
    payload["model"] = model
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("question_id", ["", " ", "\t\n"])
def test_question_ids_reject_empty_or_whitespace_names(question_id):
    payload = request_payload()
    payload["questions"] = {question_id: payload["questions"]["billing"]}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("option_id", ["", " ", "\t\n"])
def test_choice_option_ids_reject_empty_or_whitespace_names(option_id):
    payload = request_payload()
    payload["questions"] = {
        "category": {
            "type": "choice",
            "instructions": "Classify the request.",
            "criteria": {option_id: None, "valid": None},
        }
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_legacy_states_batch_field_is_rejected():
    payload = request_payload()
    payload["states"] = [{"id": "legacy", "state": "not accepted"}]
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("question_id", ["billing", "category", "urgency"])
@pytest.mark.parametrize("instructions", [None, pytest.param(..., id="missing")])
def test_instructions_are_required_and_non_null(question_id, instructions):
    payload = request_payload()
    if instructions is ...:
        payload["questions"][question_id].pop("instructions")
    else:
        payload["questions"][question_id]["instructions"] = instructions
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("count", [2, 255])
def test_choice_accepts_approved_bounds(count):
    payload = request_payload()
    payload["questions"] = {
        "choice": {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {str(index): None for index in range(count)},
        }
    }
    assert (
        len(SystemOneRequest.model_validate(payload).questions["choice"].criteria)
        == count
    )


@pytest.mark.parametrize("count", [0, 1, 256])
def test_choice_rejects_values_outside_approved_bounds(count):
    payload = request_payload()
    payload["questions"] = {
        "choice": {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {str(index): None for index in range(count)},
        }
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("count", [2, 10])
def test_score_accepts_approved_bounds(count):
    payload = request_payload()
    payload["questions"] = {
        "score": {
            "type": "score",
            "instructions": "Rate.",
            "criteria": [str(index) for index in range(count)],
        }
    }
    assert (
        len(SystemOneRequest.model_validate(payload).questions["score"].criteria)
        == count
    )


@pytest.mark.parametrize("count", [0, 1, 11])
def test_score_rejects_values_outside_approved_bounds(count):
    payload = request_payload()
    payload["questions"] = {
        "score": {
            "type": "score",
            "instructions": "Rate.",
            "criteria": [str(index) for index in range(count)],
        }
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_response_has_only_official_fields():
    request = SystemOneRequest.model_validate(request_payload())
    response = response_for(request)
    assert set(response.model_dump()) == {"model", "answers", "usage"}
    assert set(response.answers["billing"].model_dump()) == {"type", "noul"}
    with pytest.raises(ValidationError):
        SystemOneResponse.model_validate(
            {**response.model_dump(), "timing": {"inference_ms": 1.0}}
        )


def test_request_relative_response_validation_accepts_valid_math():
    request = SystemOneRequest.model_validate(request_payload())
    response = response_for(request)
    assert validate_response_for_request(request, response) is response


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "ids",
        "choice",
        "choice_confidence",
        "legend",
        "score",
        "score_confidence",
        "sum",
    ],
)
def test_request_relative_response_validation_rejects_mismatch(mutation):
    request = SystemOneRequest.model_validate(request_payload())
    body = response_for(request).model_dump()
    if mutation == "model":
        body["model"] = "llm-semantic-router/Decision-1.0-Lux-9B"
    elif mutation == "ids":
        body["answers"]["renamed"] = body["answers"].pop("billing")
    elif mutation == "choice":
        body["answers"]["category"]["choice"] = "technical"
    elif mutation == "choice_confidence":
        body["answers"]["category"]["confidence"] = 0.01
    elif mutation == "legend":
        body["answers"]["urgency"]["legend"]["2"] = "Changed"
    elif mutation == "score":
        body["answers"]["urgency"]["score"] = 1.6
    elif mutation == "score_confidence":
        body["answers"]["urgency"]["confidence"] = 0.01
    else:
        body["answers"]["category"]["probabilities"] = {
            "billing": 0.7,
            "technical": 0.2,
        }
    with pytest.raises(ResponseContractError):
        validate_response_for_request(request, SystemOneResponse.model_validate(body))


def test_confidence_matches_versioned_oracle_fixture_and_is_not_max_probability():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/decision_confidence_oracle.v1.json"
        ).read_text()
    )
    assert fixture["contract"] == "decision.normalized-top.v1"
    saw_non_max = False
    for case in fixture["cases"]:
        probabilities = case.get("probabilities")
        if probabilities is None:
            remainder = (1.0 - case["top_probability"]) / (case["option_count"] - 1)
            probabilities = [case["top_probability"]] + [remainder] * (
                case["option_count"] - 1
            )
        actual = normalized_top_confidence(probabilities)
        assert math.isclose(actual, case["expected"], abs_tol=1e-12)
        saw_non_max |= not math.isclose(actual, max(probabilities))
    assert saw_non_max


def test_batch_request_is_closed_requires_model_and_unique_bounded_state_ids():
    valid = batch_request_payload()
    valid["states"][0]["id"] = "x" * 128
    request = SystemOneBatchRequest.model_validate(valid)
    assert [state.id for state in request.states] == ["x" * 128, "second"]

    for mutation in ("missing_model", "duplicate_id", "blank_id", "long_id", "extra"):
        invalid = batch_request_payload()
        if mutation == "missing_model":
            invalid.pop("model")
        elif mutation == "duplicate_id":
            invalid["states"][1]["id"] = "first"
        elif mutation == "blank_id":
            invalid["states"][0]["id"] = " "
        elif mutation == "long_id":
            invalid["states"][0]["id"] = "x" * 129
        else:
            invalid["debug"] = True
        with pytest.raises(ValidationError):
            SystemOneBatchRequest.model_validate(invalid)


def test_batch_decision_product_accepts_1024_and_rejects_1025_or_more():
    def body(state_count, question_count):
        return {
            "model": "decision-test",
            "states": [
                {"id": f"state-{index}", "state": f"state {index}"}
                for index in range(state_count)
            ],
            "questions": {
                f"question-{index}": {
                    "type": "noul",
                    "instructions": "Is this relevant?",
                }
                for index in range(question_count)
            },
        }

    accepted = SystemOneBatchRequest.model_validate(body(32, 32))
    assert len(accepted.states) * len(accepted.questions) == MAX_BATCH_DECISIONS
    with pytest.raises(ValidationError, match="must not exceed 1024 decisions"):
        SystemOneBatchRequest.model_validate(body(41, 25))


def test_batch_response_validates_order_answer_math_and_aggregate_usage():
    request = SystemOneBatchRequest.model_validate(batch_request_payload())
    response = batch_response_for(request)
    assert validate_batch_response_for_request(request, response) is response

    reordered = response.model_copy(
        update={"results": list(reversed(response.results))}
    )
    with pytest.raises(ResponseContractError, match="identity and order"):
        validate_batch_response_for_request(request, reordered)

    wrong_usage = response.model_copy(
        update={"usage": Usage(input_tokens=999, output_tokens=6)}
    )
    with pytest.raises(ResponseContractError, match="sum of result input tokens"):
        validate_batch_response_for_request(request, wrong_usage)


def test_batch_response_forbids_diagnostic_fields():
    request = SystemOneBatchRequest.model_validate(batch_request_payload())
    body = batch_response_for(request).model_dump()
    body["timing"] = {"inference_ms": 1.0}
    with pytest.raises(ValidationError):
        SystemOneBatchResponse.model_validate(body)
