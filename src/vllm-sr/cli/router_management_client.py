"""Typed client for the Router management API."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from cli.consts import DEFAULT_API_PORT

API_ROOT = "/api/v1"
CONFIG_PATH = f"{API_ROOT}/config"
CONFIG_SCHEMA_PATH = f"{CONFIG_PATH}/schema"
RECIPES_PATH = f"{CONFIG_PATH}/recipes"
ROUTING_PREVIEW_PATH = f"{API_ROOT}/routing/preview"
OBSERVABILITY_REPLAYS_PATH = f"{API_ROOT}/observability/replays"


def default_management_base_url() -> str:
    offset = int(os.getenv("VLLM_SR_PORT_OFFSET", "0"))
    return f"http://localhost:{DEFAULT_API_PORT + offset}"


@dataclass(frozen=True)
class RouterResponse:
    payload: Any
    etag: str = ""


class RouterManagementClient:
    """Small, secret-safe client shared by agent-facing CLI workflows."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 15,
        token_env: str = "VSR_MGMT_TOKEN",
    ) -> None:
        self.base_url = _normalize_management_base_url(
            base_url or default_management_base_url()
        )
        self.timeout = timeout
        self.token_env = token_env

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        etag: str | None = None,
        params: Mapping[str, str] | None = None,
    ) -> RouterResponse:
        if not path.startswith(API_ROOT):
            raise ValueError(f"management path must start with {API_ROOT}")
        url = self.base_url + path
        headers: dict[str, str] = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if token := os.getenv(self.token_env, "").strip():
            headers["Authorization"] = f"Bearer {token}"
        if etag:
            headers["If-Match"] = etag
        try:
            response = requests.request(
                method,
                url,
                json=payload,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise ValueError(
                f"Router management API request timed out after {self.timeout:g}s"
            ) from exc
        except requests.ConnectionError as exc:
            raise ValueError(
                f"Router management API is not reachable at {self.base_url}"
            ) from exc
        except requests.RequestException as exc:
            raise ValueError("Router management API request failed") from exc

        try:
            body = response.json()
        except ValueError:
            body = response.text
        if not response.ok:
            message = _error_message(body) or str(body)[:1000]
            raise ValueError(
                f"Router management API returned HTTP {response.status_code}: {message}"
            )
        return RouterResponse(payload=body, etag=response.headers.get("ETag", ""))

    def get_config(self) -> RouterResponse:
        return self.request("GET", CONFIG_PATH)

    def get_config_schema(
        self,
        *,
        view: str = "index",
        path: str | None = None,
        surface_kind: str | None = None,
        surface_name: str | None = None,
        expanded: bool = False,
    ) -> RouterResponse:
        params = {"view": view}
        if path:
            params["path"] = path
        if surface_kind:
            params["kind"] = surface_kind
        if surface_name:
            params["name"] = surface_name
        if expanded:
            params["expanded"] = "true"
        return self.request("GET", CONFIG_SCHEMA_PATH, params=params)

    def validate_config(self, yaml_text: str) -> RouterResponse:
        return self.request(
            "POST", f"{CONFIG_PATH}/validate", payload={"yaml": yaml_text}
        )

    def plan_config(self, yaml_text: str, mode: str) -> RouterResponse:
        return self.request(
            "POST",
            f"{CONFIG_PATH}/plan",
            payload={"yaml": yaml_text, "mode": mode},
        )

    def apply_config(self, yaml_text: str, mode: str, etag: str) -> RouterResponse:
        method = {"merge": "PATCH", "replace": "PUT"}.get(mode)
        if method is None:
            raise ValueError("mode must be merge or replace")
        if not etag.strip():
            raise ValueError("config mutation requires the current ETag")
        return self.request(method, CONFIG_PATH, payload={"yaml": yaml_text}, etag=etag)

    def config_versions(self) -> RouterResponse:
        return self.request("GET", f"{CONFIG_PATH}/versions")

    def rollback_config(self, version: str, etag: str) -> RouterResponse:
        if not etag.strip():
            raise ValueError("config rollback requires the current ETag")
        return self.request(
            "POST",
            f"{CONFIG_PATH}/rollback",
            payload={"version": version},
            etag=etag,
        )

    def preview_route(
        self,
        request: dict[str, Any],
        *,
        trace: bool = False,
    ) -> RouterResponse:
        params = {"trace": "true"} if trace else None
        return self.request(
            "POST", ROUTING_PREVIEW_PATH, payload=request, params=params
        )

    def list_recipes(self) -> RouterResponse:
        return self.request("GET", RECIPES_PATH)

    def get_recipe(self, name: str) -> RouterResponse:
        return self.request("GET", f"{RECIPES_PATH}/{quote(name, safe='')}")

    def validate_recipe(self, recipe: dict[str, Any]) -> RouterResponse:
        return self.request("POST", f"{RECIPES_PATH}/validate", payload=recipe)

    def apply_recipe(
        self,
        name: str,
        recipe: dict[str, Any],
        etag: str,
    ) -> RouterResponse:
        if not etag.strip():
            raise ValueError("recipe mutation requires the current ETag")
        return self.request(
            "PUT",
            f"{RECIPES_PATH}/{quote(name, safe='')}",
            payload=recipe,
            etag=etag,
        )

    def delete_recipe(self, name: str, etag: str) -> RouterResponse:
        if not etag.strip():
            raise ValueError("recipe deletion requires the current ETag")
        return self.request(
            "DELETE",
            f"{RECIPES_PATH}/{quote(name, safe='')}",
            etag=etag,
        )


def _error_message(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or "").strip()
        message = str(error.get("message") or "").strip()
        return f"{code}: {message}" if code and message else message or code
    decision_error = str(payload.get("decision_error") or "").strip()
    if decision_error:
        policies = payload.get("applied_unknown_policies")
        if isinstance(policies, dict) and policies:
            applied = ", ".join(
                f"{key}={value}" for key, value in sorted(policies.items())
            )
            return f"{decision_error} (applied unknown policies: {applied})"
        return decision_error
    return str(payload.get("message") or "")


def _normalize_management_base_url(value: str) -> str:
    """Return an origin URL for the Router management listener."""

    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Router management endpoint must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("Router management credentials must use --token-env")
    if parsed.query or parsed.fragment:
        raise ValueError(
            "Router management endpoint must not contain query or fragment"
        )
    path = parsed.path.rstrip("/")
    if path not in {"", API_ROOT}:
        raise ValueError(f"Router management endpoint path must be empty or {API_ROOT}")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
