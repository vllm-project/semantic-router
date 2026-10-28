from __future__ import annotations

from typing import Any, ClassVar

import click
import pytest
from cli.router_management_client import (
    CONFIG_SCHEMA_PATH,
    ROUTING_PREVIEW_PATH,
    RouterManagementClient,
    default_management_base_url,
)


class _Response:
    ok = True
    status_code = 200
    headers: ClassVar[dict[str, str]] = {"ETag": '"current"'}

    def json(self) -> dict[str, bool]:
        return {"ok": True}


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8080", "http://localhost:8080/", "http://localhost:8080/api/v1"],
)
def test_management_client_builds_one_canonical_api_root(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(method: str, url: str, **kwargs: Any) -> _Response:
        calls.append((method, url, kwargs))
        return _Response()

    monkeypatch.setattr("cli.router_management_client.requests.request", request)

    response = RouterManagementClient(base_url).get_config()

    assert response.etag == '"current"'
    assert calls[0][1] == "http://localhost:8080/api/v1/config"


def test_management_client_rejects_credentials_and_arbitrary_paths() -> None:
    with pytest.raises(ValueError, match="token-env"):
        RouterManagementClient("https://user:secret@router.example")
    with pytest.raises(ValueError, match="path must be empty"):
        RouterManagementClient("https://router.example/unrelated")


def test_management_client_previews_route_with_trace_and_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(method: str, url: str, **kwargs: Any) -> _Response:
        calls.append((method, url, kwargs))
        return _Response()

    monkeypatch.setattr("cli.router_management_client.requests.request", request)
    monkeypatch.setenv("ROUTER_TOKEN", "secret")

    RouterManagementClient(
        "http://localhost:8080",
        token_env="ROUTER_TOKEN",
    ).preview_route({"text": "hello"}, trace=True)

    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == f"http://localhost:8080{ROUTING_PREVIEW_PATH}"
    assert kwargs["params"] == {"trace": "true"}
    assert kwargs["json"] == {"text": "hello"}
    assert kwargs["headers"]["Authorization"] == "Bearer secret"


def test_management_client_discovers_one_config_schema_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(method: str, url: str, **kwargs: Any) -> _Response:
        calls.append((method, url, kwargs))
        return _Response()

    monkeypatch.setattr("cli.router_management_client.requests.request", request)

    RouterManagementClient("http://localhost:8080").get_config_schema(
        view="surface",
        surface_kind="algorithm",
        surface_name="multi_factor",
    )

    method, url, kwargs = calls[0]
    assert method == "GET"
    assert url == f"http://localhost:8080{CONFIG_SCHEMA_PATH}"
    assert kwargs["params"] == {
        "view": "surface",
        "kind": "algorithm",
        "name": "multi_factor",
    }


def test_management_client_requests_expanded_section_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(method: str, url: str, **kwargs: Any) -> _Response:
        calls.append((method, url, kwargs))
        return _Response()

    monkeypatch.setattr("cli.router_management_client.requests.request", request)

    RouterManagementClient("http://localhost:8080").get_config_schema(
        view="section",
        path="routing",
        expanded=True,
    )

    assert calls[0][2]["params"] == {
        "view": "section",
        "path": "routing",
        "expanded": "true",
    }


class _UnresolvedResponse(_Response):
    ok = False
    status_code = 503

    def json(self) -> dict[str, Any]:
        return {
            "original_text": "hello",
            "decision_error": "decision unresolved",
            "applied_unknown_policies": {"guarded": "fail_request"},
        }


class _LegacyErrorResponse(_Response):
    ok = False
    status_code = 503

    def json(self) -> dict[str, Any]:
        return {"error": {"code": "CLASSIFICATION_ERROR", "message": "classifier down"}}


def test_management_client_renders_decision_unresolved_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cli.router_management_client.requests.request",
        lambda *args, **kwargs: _UnresolvedResponse(),
    )

    with pytest.raises(ValueError) as excinfo:
        RouterManagementClient("http://localhost:8080").preview_route({"text": "hello"})

    message = str(excinfo.value)
    assert "503" in message
    assert "decision unresolved" in message
    assert "guarded=fail_request" in message


def test_management_client_keeps_legacy_503_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cli.router_management_client.requests.request",
        lambda *args, **kwargs: _LegacyErrorResponse(),
    )

    with pytest.raises(ValueError, match="503: CLASSIFICATION_ERROR: classifier down"):
        RouterManagementClient("http://localhost:8080").preview_route({"text": "hello"})


def test_default_management_base_url_applies_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "30")

    assert default_management_base_url() == "http://localhost:8110"


def test_default_management_base_url_rejects_non_numeric_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "abc")

    with pytest.raises(
        click.ClickException,
        match="must be an integer between 0 and 15484",
    ):
        default_management_base_url()


def test_default_management_base_url_rejects_negative_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "-1")

    with pytest.raises(
        click.ClickException,
        match="must be between 0 and 15484 so derived ports stay within 65535, got -1",
    ):
        default_management_base_url()


def test_default_management_base_url_accepts_the_largest_useful_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "15484")

    assert default_management_base_url() == "http://localhost:23564"


def test_default_management_base_url_rejects_offset_that_overflows_a_derived_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "15485")

    with pytest.raises(
        click.ClickException,
        match="must be between 0 and 15484 so derived ports stay within 65535",
    ):
        default_management_base_url()


def test_default_management_base_url_rejects_offset_above_the_derived_port_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "57456")

    with pytest.raises(
        click.ClickException,
        match="must be between 0 and 15484 so derived ports stay within 65535, got 57456",
    ):
        default_management_base_url()


def _recorded_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, dict[str, Any]]]:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(method: str, url: str, **kwargs: Any) -> _Response:
        calls.append((method, url, kwargs))
        return _Response()

    monkeypatch.setattr("cli.router_management_client.requests.request", request)
    return calls


def test_apply_config_sends_a_merge_as_patch_with_the_if_match_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _recorded_requests(monkeypatch)

    RouterManagementClient("http://localhost:8080").apply_config(
        "version: v0.3\n", "merge", '"v7"'
    )

    method, url, kwargs = calls[0]
    assert method == "PATCH"
    assert url == "http://localhost:8080/api/v1/config"
    assert kwargs["headers"]["If-Match"] == '"v7"'
    assert kwargs["json"] == {"yaml": "version: v0.3\n"}


def test_apply_config_sends_a_replace_as_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _recorded_requests(monkeypatch)

    RouterManagementClient("http://localhost:8080").apply_config(
        "version: v0.3\n", "replace", '"v7"'
    )

    assert calls[0][0] == "PUT"
    assert calls[0][2]["headers"]["If-Match"] == '"v7"'


def test_apply_config_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be merge or replace"):
        RouterManagementClient("http://localhost:8080").apply_config(
            "version: v0.3\n", "reset", '"v7"'
        )


def test_apply_config_rejects_an_etag_that_is_blank() -> None:
    client = RouterManagementClient("http://localhost:8080")

    for blank in ("", "   "):
        with pytest.raises(ValueError, match="requires the current ETag"):
            client.apply_config("version: v0.3\n", "merge", blank)


def test_rollback_config_submits_the_version_with_the_if_match_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _recorded_requests(monkeypatch)

    RouterManagementClient("http://localhost:8080").rollback_config("3", '"v7"')

    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == "http://localhost:8080/api/v1/config/rollback"
    assert kwargs["json"] == {"version": "3"}
    assert kwargs["headers"]["If-Match"] == '"v7"'


def test_rollback_config_rejects_an_etag_that_is_blank() -> None:
    with pytest.raises(ValueError, match="requires the current ETag"):
        RouterManagementClient("http://localhost:8080").rollback_config("3", " ")


def test_plan_config_posts_the_document_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _recorded_requests(monkeypatch)

    RouterManagementClient("http://localhost:8080").plan_config(
        "version: v0.3\n", "merge"
    )

    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == "http://localhost:8080/api/v1/config/plan"
    assert kwargs["json"] == {"yaml": "version: v0.3\n", "mode": "merge"}


def test_config_versions_reads_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _recorded_requests(monkeypatch)

    RouterManagementClient("http://localhost:8080").config_versions()

    assert calls[0][0] == "GET"
    assert calls[0][1] == "http://localhost:8080/api/v1/config/versions"


class _PreconditionFailedResponse(_Response):
    ok = False
    status_code = 412

    def json(self) -> dict[str, Any]:
        return {
            "error": {
                "code": "PRECONDITION_FAILED",
                "message": "the configuration changed since the ETag was read",
            }
        }


def test_a_stale_etag_answers_precondition_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cli.router_management_client.requests.request",
        lambda *args, **kwargs: _PreconditionFailedResponse(),
    )

    with pytest.raises(ValueError, match="412: PRECONDITION_FAILED") as excinfo:
        RouterManagementClient("http://localhost:8080").apply_config(
            "version: v0.3\n", "merge", '"old"'
        )

    error = excinfo.value
    assert getattr(error, "status", None) == 412
    assert getattr(error, "code", None) == "PRECONDITION_FAILED"
    assert "changed since the ETag" in getattr(error, "detail", "")
