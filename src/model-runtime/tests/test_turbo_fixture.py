"""The ``turbo`` fixture serves d3-turbo's System One contract (3.2.0): a state-once card with Box and track,
question and criteria images, the validated length, the per-request cap and the video forms' clip limit.
"""

import asyncio

import pytest

from .conftest import start_runtime


def call(runtime, body):
    return asyncio.run(runtime.call("decisions", body))


@pytest.fixture(scope="module")
def turbo(fixture_root):
    from vllm_srun.testing.decision3 import TURBO_VALIDATED_LENGTH, write_fixture

    runtime = start_runtime(write_fixture(fixture_root / "d3-turbo", "turbo", seed=21))
    yield runtime, TURBO_VALIDATED_LENGTH
    runtime.stop()


def test_the_card_is_d3_turbos(turbo):
    runtime, validated = turbo
    card = runtime.lookup(None).card([])
    assert card["usage"] == "state_once"
    assert {"choice", "noul", "score", "box", "track"} <= set(card["question_types"])
    assert card["image_placements"] == ["state", "instructions", "criteria"]
    limits = card["limits"]
    assert limits["validated_input_tokens"] == validated < limits["max_input_tokens"]
    assert limits["max_request_tokens"] == limits["max_input_tokens"]
    assert limits["box_video_max_seconds"] == 60
    assert limits["video_max_seconds"] >= limits["box_video_max_seconds"]


def test_box_and_question_images_are_answered(turbo, png_url):
    runtime, _ = turbo
    photo, swatch = (
        png_url(96, 64, 3).split(",", 1)[1],
        png_url(32, 32, 4).split(",", 1)[1],
    )
    status, body = call(
        runtime,
        {
            "state": {"photo": "<photo>"},
            "assets": {
                "<photo>": {"data_base64": photo},
                "<swatch>": {"data_base64": swatch},
            },
            "questions": {
                "where": {"type": "box", "instructions": "the brightest patch"},
                "match": {
                    "type": "choice",
                    "instructions": "Which colour is closest to <swatch>?",
                    "criteria": {"warm": "a warm colour", "cool": "a cool colour"},
                },
                "pick": {
                    "type": "choice",
                    "instructions": "Which looks like the photo?",
                    "criteria": {"a": "<swatch>", "b": "plain grey"},
                },
            },
        },
    )
    assert status == 200, body
    answers = body["answers"]
    box = answers["where"]
    assert box["type"] == "box" and set(box) == {"type", "box", "confidence"}, box
    edges = box["box"]
    assert (
        0 <= edges["top"] < edges["bottom"] <= 1
        and 0 <= edges["left"] < edges["right"] <= 1
    )
    assert box["confidence"] is None or 0 <= box["confidence"] <= 1
    assert answers["match"]["choice"] in {"warm", "cool"}, answers["match"]
    assert answers["pick"]["choice"] in {"a", "b"}, answers["pick"]


def test_usage_counts_the_state_once(turbo):
    runtime, _ = turbo
    state = "w " * 150
    one = {"q1": {"type": "noul", "instructions": "Is it long?"}}
    two = {**one, "q2": {"type": "noul", "instructions": "Is it long?"}}
    _, first = call(runtime, {"state": state, "questions": one})
    _, both = call(runtime, {"state": state, "questions": two})
    assert "error" not in first["answers"]["q1"], first
    question = both["usage"]["input_tokens"] - first["usage"]["input_tokens"]
    assert 0 < question < first["usage"]["input_tokens"] / 3, (
        first["usage"],
        both["usage"],
    )


def test_long_context_and_the_request_cap(turbo):
    runtime, validated = turbo
    card = runtime.lookup(None).card([])
    body = {
        "state": "word " * validated,
        "questions": {"q": {"type": "noul", "instructions": "Is it long?"}},
    }
    status, answer = call(runtime, body)
    assert status == 200 and answer["answers"]["q"]["reason"] == "question_tokens"
    status, answer = call(runtime, {**body, "options": {"long_context": True}})
    assert status == 200 and 0 <= answer["answers"]["q"]["noul"] <= 1
    cap = card["limits"]["max_request_tokens"]
    many = {
        f"q{i}": {"type": "noul", "instructions": "word " * 2000}
        for i in range(cap // 2000 + 1)
    }
    status, answer = call(
        runtime,
        {"state": "short", "questions": many, "options": {"long_context": True}},
    )
    assert status == 400 and answer["error"]["reason"] == "request_tokens", answer
    assert answer["error"]["limit"] == cap


def test_video_forms_need_a_video(turbo, png_url):
    runtime, _ = turbo
    status, body = call(
        runtime,
        {
            "images": [png_url(64, 48, 5)],
            "state": "a still",
            "questions": {
                "track": {"type": "track", "instructions": "the dog", "rate": 2},
                "at": {"type": "box", "instructions": "the dog", "at": 1.0},
                "rate": {"type": "track", "instructions": "the dog", "rate": 3},
                "still": {"type": "noul", "instructions": "Is it a still?"},
            },
        },
    )
    assert status == 200, body
    reasons = {key: answer.get("reason") for key, answer in body["answers"].items()}
    assert reasons == {
        "track": "video_needed",
        "at": "video_needed",
        "rate": "question_malformed",
        "still": None,
    }, reasons
