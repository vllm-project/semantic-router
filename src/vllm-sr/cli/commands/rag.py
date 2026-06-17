"""RAG command implementations (#2117).

`vllm-sr rag list` lists the vector stores created through the RAG
ingestion pipeline (the OpenAI-compatible Vector Stores API added in the
#1262 work). It calls the router's ``GET /v1/vector_stores`` endpoint on
the router API port (default 8080, like `vllm-sr eval`) — not the Envoy /
OpenAI listener, which does not serve the management API.
"""

from __future__ import annotations

import os
from http import HTTPStatus
from typing import Any
from urllib.parse import urljoin

import requests

from cli.consts import DEFAULT_API_PORT
from cli.utils import get_logger

log = get_logger(__name__)

VECTOR_STORES_PATH = "/v1/vector_stores"
_HTTP_FORBIDDEN = 403


def _default_endpoint() -> str:
    # Support port offset via environment variable (set during vllm-sr serve).
    port_offset = int(os.getenv("VLLM_SR_PORT_OFFSET", "0"))
    api_port = DEFAULT_API_PORT + port_offset
    return f"http://localhost:{api_port}{VECTOR_STORES_PATH}"


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize so we always end up calling /v1/vector_stores."""
    endpoint = endpoint.strip()
    if not endpoint:
        return _default_endpoint()

    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]

    if endpoint.endswith(VECTOR_STORES_PATH):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/vector_stores"

    return urljoin(endpoint + "/", VECTOR_STORES_PATH.lstrip("/"))


def rag_list_command(endpoint: str | None = None, timeout: int = 15) -> None:
    """List vector stores from the router's Vector Stores API.

    Args:
        endpoint: Router base URL or full /v1/vector_stores endpoint. Defaults
            to the local router API port (8080 + VLLM_SR_PORT_OFFSET).
        timeout: HTTP request timeout in seconds.
    """
    url = _normalize_endpoint(endpoint or "")

    try:
        resp = requests.get(url, timeout=timeout)
    except requests.ConnectionError as exc:
        raise ValueError(
            f"Router is not running at {url}. Start the router with "
            "'vllm-sr serve' and retry."
        ) from exc
    except requests.Timeout as exc:
        raise ValueError(
            f"Request to {url} timed out after {timeout}s. Is the router healthy?"
        ) from exc
    except requests.RequestException as exc:
        raise ValueError(
            f"Failed to call router vector stores endpoint {url}: {exc}"
        ) from exc

    if resp.status_code == HTTPStatus.SERVICE_UNAVAILABLE:
        raise ValueError(
            "Vector store feature is not enabled on the router. Enable a vector "
            "store backend in your config to use RAG ingestion and retrieval."
        )
    if resp.status_code != requests.codes.ok:
        raise ValueError(_format_error_response(resp, url))

    try:
        data = resp.json()
    except ValueError as exc:
        raise ValueError(f"Router returned non-JSON response: {resp.text}") from exc

    stores = data.get("data") if isinstance(data, dict) else None
    if not isinstance(stores, list):
        stores = []

    log.info("=" * 60)
    log.info("vLLM Semantic Router - Vector Stores")
    log.info("=" * 60)
    log.info(f"Endpoint: {url}")
    log.info("")
    _print_vector_stores(stores)


def _format_error_response(resp: Any, url: str) -> str:
    """Extract a clean message from a non-200 router response."""
    try:
        body = resp.json()
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
            code = err.get("code", "")
            message = err.get("message", "")
            if code and message:
                return f"Router returned {resp.status_code} {code}: {message}"
            if message:
                return f"Router returned {resp.status_code}: {message}"
    except ValueError:
        pass

    content_type = getattr(resp, "headers", {}).get("Content-Type", "")
    if resp.status_code == _HTTP_FORBIDDEN and "text/html" in content_type:
        return (
            f"Router returned 403 from {url}. This looks like a proxy or gateway "
            "— point --endpoint at the router API port (default: 8080), not Envoy."
        )
    return f"Router vector stores request failed: HTTP {resp.status_code} - {resp.text}"


def _print_vector_stores(stores: list) -> None:
    log.info(f"Vector stores ({len(stores)}):")
    if not stores:
        log.info("  (none created)")
        return

    for store in stores:
        if not isinstance(store, dict):
            continue
        name = str(store.get("name") or "(unnamed)")
        store_id = str(store.get("id") or "-")
        log.info(f"  - {name}  ({store_id})")

        status = store.get("status")
        if status:
            log.info(f"      status:       {status}")

        backend = store.get("backend_type")
        if backend:
            log.info(f"      backend:      {backend}")

        counts = store.get("file_counts")
        if isinstance(counts, dict):
            total = counts.get("total", 0)
            completed = counts.get("completed", 0)
            in_progress = counts.get("in_progress", 0)
            failed = counts.get("failed", 0)
            log.info(
                f"      files:        {total} total "
                f"({completed} completed, {in_progress} in progress, {failed} failed)"
            )
