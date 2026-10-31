"""Text-only NLI must refuse attachments instead of proving partial input complete."""

import pytest

from . import test_nli
from .test_api_contract import check
from .test_nli import QUESTIONS, STATE

packages = test_nli.packages
runtime = test_nli.runtime
client = test_nli.client

PATHS = ("/v1/decisions", "/v1/systemone", "/v1/bundle")
ATTACHMENTS = {
    "images": ["data:image/png;base64,AAAA"],
    "videos": ["data:video/mp4;base64,AAAA"],
}


def body(strict=False):
    questions = {
        key: {**question, "require_full_input": True} if strict else question
        for key, question in QUESTIONS.items()
    }
    return {"model": "nli", "state": STATE, "questions": questions}


def with_media(strict, named, media):
    request = body(strict)
    if named:
        request["states"] = {
            "attached": {"state": STATE, "questions": request["questions"], **media}
        }
    else:
        request.update(media)
    return request


def post(client, path, request):
    if path == "/v1/bundle":
        response = client.post(
            path,
            json={
                "tasks": [
                    {"id": "attached", "decisions": request},
                    {"id": "text", "decisions": body(True)},
                ]
            },
        )
        assert response.status_code == 200, response.text
        check("BundleResponse", response.json())
        attached, text = response.json()["results"]
        assert text["status"] == 200
        assert all(
            answer["input_coverage"] == "complete"
            for answer in text["decisions"]["answers"].values()
        )
        return attached["status"], attached.get("decisions", attached)
    response = client.post(path, json=request)
    check(
        "DecisionResponse" if response.status_code == 200 else "ErrorResponse",
        response.json(),
    )
    return response.status_code, response.json()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("named", [False, True], ids=["root", "named-state"])
@pytest.mark.parametrize("strict", [False, True], ids=["ordinary", "strict"])
@pytest.mark.parametrize("field", ATTACHMENTS)
def test_unsupported_media_is_rejected(
    client, runtime, monkeypatch, path, named, strict, field
):
    if not named and path != "/v1/bundle":
        model = runtime.lookup("nli").model

        def unexpected_prepare(*args, **kwargs):
            raise AssertionError("unsupported media reached hypothesis preparation")

        monkeypatch.setattr(model.head, "prepare", unexpected_prepare)
    status, response = post(
        client, path, with_media(strict, named, {field: ATTACHMENTS[field]})
    )
    assert status == 400, response
    assert response["error"]["code"] == "invalid_request"
    assert field in response["error"]["message"]
    assert "answers" not in response and "input_coverage" not in response


@pytest.mark.parametrize("field", ATTACHMENTS)
@pytest.mark.parametrize("value", ["not-a-list", {}, 0, False])
def test_malformed_media_fields_are_rejected(client, field, value):
    status, response = post(
        client, "/v1/decisions", with_media(True, False, {field: value})
    )
    assert status == 400, response
    assert response["error"]["code"] == "invalid_request"
    assert field in response["error"]["message"]
    assert "answers" not in response


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("named", [False, True], ids=["root", "named-state"])
@pytest.mark.parametrize("strict", [False, True], ids=["ordinary", "strict"])
@pytest.mark.parametrize(
    "media", [{}, {"images": [], "videos": []}, {"images": None, "videos": None}]
)
def test_text_only_requests_keep_answers_and_coverage(
    client, path, named, strict, media
):
    _, expected = post(client, path, with_media(strict, named, {}))
    status, actual = post(client, path, with_media(strict, named, media))
    assert status == 200, actual
    assert actual == expected
    answers = [*actual["answers"].values()]
    if named:
        answers.extend(actual["states"]["attached"]["answers"].values())
    assert all(("input_coverage" in answer) == strict for answer in answers)
