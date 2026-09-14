from __future__ import annotations

from typing import Any, ClassVar

import pytest
from cli.router_management_client import (
    CONFIG_SCHEMA_PATH,
    ROUTING_PREVIEW_PATH,
    RouterManagementClient,
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
