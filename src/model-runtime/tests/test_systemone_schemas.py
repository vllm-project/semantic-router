"""The System One schemas a runtime speaks: vllm-sr/5.0 and the vllm-sr/4.0 superset, read into the internal form."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vllm_srun.api import schemas
from vllm_srun.families.vela2.request import read_state
from vllm_srun.families.vela3_d3 import spans
from vllm_srun.systemone import (
    IMPLICIT_ROLES,
    REQUEST_ROLES,
    implicit_roles,
    normalize_presets,
    read_roles,
)

FIVE, FOUR = schemas.SCHEMA_50, schemas.SCHEMA_40
NOUL = {"q": {"type": "noul", "instructions": "Supported?"}}
GROUNDED = {"context": "Refunds within 30 days.", "answer": "You have 90 days."}
# The cases the Router's Go tests share (config/schemas/systemone/implicit-roles.json).
IMPLICIT_CASES = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "config/schemas/systemone/implicit-roles.json"
    ).read_text(encoding="utf-8")
)["cases"]
# The requests without a valid question the Router's Go tests share
# (config/schemas/systemone/no-valid-question.json).
NO_VALID_QUESTION = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "config/schemas/systemone/no-valid-question.json"
    ).read_text(encoding="utf-8")
)["cases"]


def read(schema, path, body, headers=None):
    return schemas.read(schema, path, body, headers or {})


def test_a_five_request_is_read_strictly_into_the_internal_form() -> None:
    internal = read(
        FIVE,
        schemas.PATH_SYSTEMONE,
        {"model": "m", "state": GROUNDED, "questions": NOUL},
    ).body
    assert internal == {
        "model": "m",
        "state": GROUNDED,
        "questions": NOUL,
        "roles": None,
    }
    text = read(
        FIVE, schemas.PATH_SYSTEMONE, {"state": "plain", "questions": NOUL}
    ).body
    assert "roles" not in text
    for body, message in (
        (
            {
                "state": "x",
                "questions": {"q": {"type": "choice", "choices": [{"key": "a"}]}},
            },
            "use criteria",
        ),
        ({"state": "x", "images": [], "questions": NOUL}, "use assets"),
        (
            {"states": {"a": {"state": "x", "questions": NOUL}}},
            "use POST /v1/systemone/batch",
        ),
        (
            {"state": "x", "questions": NOUL, "options": {"long_context": True}},
            "TypeSafe-Danger-Zone-Long-Context",
        ),
        (
            {"state": "x", "questions": {"q": {"type": "noul", "colour": "red"}}},
            "colour is not a field",
        ),
        ({"state": "x", "questions": {"q": {"preset": "pii"}}}, "type is required"),
        (
            {"state": "x", "roles": {"x": "user"}, "questions": NOUL},
            "roles name the keys of an object state",
        ),
        (
            {"state": {"a": "x"}, "roles": {"b": "user"}, "questions": NOUL},
            "roles.b names no key",
        ),
    ):
        with pytest.raises(schemas.ReadError, match=message):
            read(FIVE, schemas.PATH_SYSTEMONE, body)


def test_media_leading_a_five_state_is_read_in_front_of_it() -> None:
    body = {
        "state": ["img", "clip", {"request": "What happens?"}],
        "questions": NOUL,
        "assets": {
            "img": {"type": "image", "format": "png", "data_base64": "iVBORw0KGgo="},
            "clip": {"type": "video", "format": "mp4", "data_base64": "AAAA"},
        },
    }
    internal = read(
        FIVE, schemas.PATH_SYSTEMONE, body, {schemas.LONG_CONTEXT_HEADER: "true"}
    ).body
    assert internal["state"] == {"request": "What happens?"}
    assert internal["images"] == ["data:image/png;base64,iVBORw0KGgo="]
    assert internal["videos"] == ["data:video/mp4;base64,AAAA"]
    assert "assets" not in internal and internal["options"] == {"long_context": True}
    inline = {
        "state": ["look at", "img"],
        "questions": NOUL,
        "assets": {"img": body["assets"]["img"]},
    }
    assert read(FIVE, schemas.PATH_SYSTEMONE, inline).body["assets"] == inline["assets"]


def test_a_batch_is_named_requests_on_either_schema() -> None:
    body = {
        "model": "m",
        "requests": {
            "a": {"state": "one", "questions": NOUL},
            "b": {"model": "m", "state": GROUNDED, "questions": NOUL},
        },
        "options": {"deadline_ms": 800},
    }
    five = read(FIVE, schemas.PATH_BATCH, body)
    assert five.names == ["a", "b"]
    assert five.body == {
        "model": "m",
        "states": {
            "a": {"state": "one", "questions": NOUL},
            "b": {"state": GROUNDED, "questions": NOUL, "roles": None},
        },
        "options": {"deadline_ms": 800},
    }
    four = read(FOUR, schemas.PATH_BATCH, body)
    assert "roles" not in four.body["states"]["b"]
    with pytest.raises(schemas.ReadError, match="is not the batch's model"):
        read(
            FIVE,
            schemas.PATH_BATCH,
            {
                "model": "m",
                "requests": {"a": {"model": "n", "state": "x", "questions": NOUL}},
            },
        )
    answered = schemas.answer(
        five,
        {"model": "m", "states": {"a": {"answers": {}}}, "usage": {"input_tokens": 1}},
    )
    assert answered == {
        "model": "m",
        "responses": {"a": {"answers": {}}},
        "usage": {"input_tokens": 1},
    }


def test_a_four_runtime_reads_every_request_as_it_came() -> None:
    body = {
        "state": {"request": "x"},
        "questions": {"q": {"type": "choice", "choices": [{"key": "a"}, {"key": "b"}]}},
        "images": [],
    }
    assert read(FOUR, schemas.PATH_SYSTEMONE, body).body is body
    assert read(FIVE, schemas.PATH_DECISIONS, body).body is body


def test_errors_follow_the_clients_schema() -> None:
    native = {
        "error": {
            "code": "question_failed",
            "message": "question 'q' cannot be answered",
            "reason": "question_malformed",
            "question": "q",
            "state": "b",
            "question_error": "invalid_question",
        }
    }
    status, body, _ = schemas.error(FOUR, schemas.PATH_SYSTEMONE, 422, native)
    assert (
        status == 422
        and body["error"] == native["error"]
        and body["detail"] == "question 'q' cannot be answered"
    )
    assert (
        body["code"] == "question_failed"
        and body["question"] == "q"
        and body["state"] == "b"
    )
    status, body, _ = schemas.error(FIVE, schemas.PATH_BATCH, 422, native, batch=True)
    assert status == 422 and "error" not in body
    assert body["detail"] == [
        {
            "loc": ["body", "requests", "b", "questions", "q"],
            "msg": "question 'q' cannot be answered",
            "type": "invalid_question",
        }
    ]
    status, body, headers = schemas.error(
        FIVE,
        schemas.PATH_SYSTEMONE,
        429,
        {"error": {"code": "overloaded", "message": "busy"}},
    )
    assert (status, body, headers) == (
        529,
        {"detail": "busy", "code": "overloaded"},
        {"Retry-After": "1"},
    )
    status, body, _ = schemas.error(
        FIVE,
        schemas.PATH_DECISIONS,
        400,
        {"error": {"code": "invalid_request", "message": "bad"}},
    )
    assert (
        status == 400
        and body["error"]["code"] == "invalid_request"
        and body["detail"] == "bad"
    )


def test_the_five_listing_carries_version_features_and_card() -> None:
    cards = [{"id": "m", "question_types": ["choice", "noul"]}]
    listed = [{"name": "m", "description": "d", "release_date": ""}]
    assert schemas.listing(FIVE, cards, listed) == [
        {
            "name": "m",
            "description": "d",
            "release_date": "",
            "version": "5.0",
            "features": ["choice", "noul"],
            "card": cards[0],
        }
    ]
    assert schemas.listing(FOUR, cards, listed) == listed


def typed(state, roles):
    token = REQUEST_ROLES.set(roles)
    try:
        return read_state(state)
    finally:
        REQUEST_ROLES.reset(token)


def test_roles_that_name_what_the_keys_imply_read_the_state_as_before() -> None:
    state = {
        "request": "Again, the export.",
        "history": "How do I export?",
        "previous_answer": "Use the menu.",
        "answer": "Retry.",
    }
    implicit = typed(state, IMPLICIT_ROLES)
    explicit = typed(
        state,
        {
            "request": "user",
            "history": "context",
            "previous_answer": "context",
            "answer": "answer",
        },
    )
    assert (explicit.parts, explicit.fields) == (implicit.parts, implicit.fields)


@pytest.mark.parametrize("case", IMPLICIT_CASES)
def test_roles_a_router_writes_for_a_four_request_read_as_it_was_written(case) -> None:
    """The Router gives a 4.0 request's keys the roles the runtime's implicit roles give them, whatever their case;
    roles equal to them are the implicit roles, so every family (a vela3_d3 span's target too) reads the state as before.
    """
    state, roles = case["state"], case["roles"]
    assert implicit_roles(state) == roles
    read = read_roles({"state": state, "roles": roles})
    assert read is IMPLICIT_ROLES
    implicit, explicit = typed(state, IMPLICIT_ROLES), typed(state, read)
    assert (explicit.parts, explicit.fields) == (implicit.parts, implicit.fields)
    token = REQUEST_ROLES.set(read)
    try:
        target = spans.default_target(list(state))
    finally:
        REQUEST_ROLES.reset(token)
    assert target == ("answer" if "answer" in state else list(state)[-1])


def test_roles_other_than_the_implicit_ones_stay_explicit() -> None:
    state = {"Prompt": "Is the sky green?", "Response": "Yes."}
    roles = {"Prompt": "context", "Response": "answer"}
    assert read_roles({"state": state, "roles": roles}) == roles
    assert read_roles({"state": state, "roles": None}) is None


def test_a_five_state_without_roles_is_the_users_content_as_documented() -> None:
    """The documented difference: {context, answer} is a context and an answer part under vllm-sr/4.0's implicit
    roles, and one user part holding both keys under vllm-sr/5.0 without roles."""
    four = typed(
        GROUNDED,
        read_roles(
            read(
                FOUR, schemas.PATH_SYSTEMONE, {"state": GROUNDED, "questions": NOUL}
            ).body
        ),
    )
    five = typed(
        GROUNDED,
        read_roles(
            read(
                FIVE, schemas.PATH_SYSTEMONE, {"state": GROUNDED, "questions": NOUL}
            ).body
        ),
    )
    assert [part.role for part in four.parts] == ["context", "answer"]
    assert [part.role for part in five.parts] == ["user"]
    assert (
        five.text("user")
        == "context:\nRefunds within 30 days.\n\nanswer:\nYou have 90 days."
    )
    assert (
        four.text("context") == "Refunds within 30 days."
        and four.text("answer") == "You have 90 days."
    )


def test_a_span_reads_the_answer_its_roles_name() -> None:
    token = REQUEST_ROLES.set({"claim": "answer", "notes": "context"})
    try:
        assert spans.default_target(["claim", "notes"]) == "claim"
    finally:
        REQUEST_ROLES.reset(token)
    assert spans.default_target(["request", "answer"]) == "answer"
    token = REQUEST_ROLES.set(None)
    try:
        assert spans.parts_of({"a": "x", "b": "y"}) == [
            ("user", '{"a": "x", "b": "y"}')
        ]
    finally:
        REQUEST_ROLES.reset(token)


def test_a_five_preset_question_asks_the_trained_preset_when_the_model_has_one() -> (
    None
):
    questions = {
        "pii": {
            "type": "span",
            "instructions": "Find personal data.",
            "criteria": {"EMAIL": None},
            "preset": "pii",
            "threshold": 0.4,
        },
        "jailbreak": {
            "type": "choice",
            "instructions": "Attack?",
            "criteria": {"yes": None, "no": None},
            "preset": "jailbreak",
        },
        "legacy": {"preset": "halu"},
    }
    normalized = normalize_presets(questions, ("pii", "halu"))
    assert normalized["pii"] == {"preset": "pii", "threshold": 0.4}
    assert normalized["jailbreak"] == {
        "type": "choice",
        "instructions": "Attack?",
        "criteria": {"yes": None, "no": None},
    }
    assert normalized["legacy"] is questions["legacy"]


@pytest.fixture(scope="module")
def vela2_runtimes(tmp_path_factory):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime
    from vllm_srun.testing.vela2 import write_encoder_package

    package = write_encoder_package(tmp_path_factory.mktemp("schemas") / "vela2")
    started = {}
    for schema in (FIVE, FOUR):
        runtime = Runtime(
            ServeConfig(
                models=(ModelConfig(model=str(package), device="cpu"),),
                systemone_schema=schema,
            )
        )
        runtime.start(background=False)
        started[schema] = runtime
    yield started
    for runtime in started.values():
        runtime.stop()


def post(runtime, path, body, headers=None):
    from starlette.testclient import TestClient
    from vllm_srun.api.app import create_app

    with TestClient(create_app(runtime)) as client:
        response = client.post(path, json=body, headers=headers or {})
    return response.status_code, response.json(), response.headers


def test_a_five_runtime_and_a_four_runtime_read_the_same_object_state_differently(
    vela2_runtimes,
) -> None:
    body = {"state": GROUNDED, "questions": NOUL}
    five = post(vela2_runtimes[FIVE], "/v1/systemone", body)
    four = post(vela2_runtimes[FOUR], "/v1/systemone", body)
    assert five[0] == four[0] == 200
    assert five[1]["usage"]["input_tokens"] != four[1]["usage"]["input_tokens"]
    roles = {"context": "context", "answer": "answer"}
    explicit = post(vela2_runtimes[FIVE], "/v1/systemone", {**body, "roles": roles})
    assert (
        explicit[1]["answers"] == four[1]["answers"]
        and explicit[1]["usage"] == four[1]["usage"]
    )


def test_a_runtime_answers_the_batch_endpoint_and_shapes_its_errors(
    vela2_runtimes,
) -> None:
    status, body, _ = post(
        vela2_runtimes[FIVE],
        "/v1/systemone/batch",
        {
            "model": vela2_runtimes[FIVE].served[0].served_id,
            "requests": {"a": {"state": "one", "questions": NOUL}},
        },
    )
    assert status == 200 and set(body["responses"]) == {"a"} and "states" not in body
    status, body, _ = post(
        vela2_runtimes[FIVE],
        "/v1/systemone",
        {"state": "x", "questions": {"q": {"type": "noul", "colour": "red"}}},
    )
    assert status == 400 and "error" not in body and "colour" in body["detail"]
    status, body, _ = post(
        vela2_runtimes[FOUR],
        "/v1/systemone",
        {"state": "x", "questions": NOUL, "colour": "red"},
    )
    assert (
        status == 400 and body["error"]["code"] == "invalid_request" and body["detail"]
    )


@pytest.mark.parametrize("schema", [FIVE, FOUR])
@pytest.mark.parametrize("case", NO_VALID_QUESTION, ids=lambda case: case["name"])
def test_a_request_without_a_valid_question_answers_as_the_router_answers_it(
    vela2_runtimes, schema, case
) -> None:
    status, body, _ = post(vela2_runtimes[schema], "/v1/decisions", case["request"])
    assert status == case["status"], body
    if "error" in case:
        assert {key: body["error"].get(key) for key in case["error"]} == case[
            "error"
        ], body
    for name, failed in case.get("failed", {}).items():
        for question, reason in failed.items():
            answer = body["states"][name]["answers"][question]
            assert (answer["error"], answer.get("reason")) == (
                "invalid_question",
                reason,
            ), body
