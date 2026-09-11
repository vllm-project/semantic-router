"""Tests for ``vllm-sr storage vector-stores``.

The command lists vector stores from the router's ``GET /api/v1/storage/vector-stores``
endpoint (the RAG ingestion / Vector Stores API), served on the router API
port (default 8080), not the Envoy listener. It must:
- Be discoverable from the top-level CLI help.
- Print each vector store's name, id, status, backend, and file counts.
- Report "(none created)" when the API returns an empty list.
- Surface a clear message when the vector store feature is disabled (503).
- Surface a clear message when the router is unreachable.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
vector_store_commands = importlib.import_module("cli.commands.vector_stores")
management_client = importlib.import_module("cli.router_management_client")


def _fake_response(status_code: int, json_body=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.text = text
    resp.headers = {}
    if json_body is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = json_body
    return resp


_TWO_STORES = {
    "object": "list",
    "data": [
        {
            "id": "vs_abc123",
            "object": "vector_store",
            "name": "docs-prod",
            "status": "active",
            "backend_type": "milvus",
            "file_counts": {
                "in_progress": 1,
                "completed": 9,
                "failed": 0,
                "total": 10,
            },
        },
        {
            "id": "vs_def456",
            "object": "vector_store",
            "name": "support-kb",
            "status": "active",
            "backend_type": "memory",
            "file_counts": {
                "in_progress": 0,
                "completed": 3,
                "failed": 1,
                "total": 4,
            },
        },
    ],
}


def test_rag_list_registered_on_top_level_help():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "storage" in result.output


def test_rag_list_help_describes_subcommand():
    runner = CliRunner()

    result = runner.invoke(main, ["storage", "--help"])

    assert result.exit_code == 0
    assert "vector-stores" in result.output


def test_rag_list_prints_vector_stores(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, _TWO_STORES),
    )

    result = runner.invoke(main, ["storage", "vector-stores"])

    assert result.exit_code == 0
    combined = result.stdout
    assert result.stderr == ""

    assert "Vector stores (2)" in combined
    assert "docs-prod" in combined
    assert "vs_abc123" in combined
    assert "support-kb" in combined
    assert "milvus" in combined
    assert "10 total" in combined
    assert "1 failed" in combined


def test_rag_list_reports_no_vector_stores(monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(200, {"object": "list", "data": []}),
    )

    result = runner.invoke(main, ["storage", "vector-stores"])

    assert result.exit_code == 0
    combined = result.stdout
    assert result.stderr == ""
    assert "Vector stores (0)" in combined
    assert "No vector stores have been created" in combined


def test_rag_list_reports_feature_disabled(caplog, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        management_client.requests,
        "request",
        lambda *a, **k: _fake_response(503, text="disabled"),
    )

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["storage", "vector-stores"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "HTTP 503" in combined


def test_rag_list_reports_unreachable_router(caplog, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    def _boom(*a, **k):
        raise management_client.requests.ConnectionError("refused")

    monkeypatch.setattr(management_client.requests, "request", _boom)

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["storage", "vector-stores"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "Router management API is not reachable at http://localhost:8080" in combined


def test_rag_list_rejects_endpoint_credentials():
    result = CliRunner().invoke(
        main,
        [
            "storage",
            "vector-stores",
            "--endpoint",
            "http://user:RAGSECRET@router.test",
        ],
    )

    assert result.exit_code == 1
    assert "RAGSECRET" not in result.stdout


def test_rag_list_uses_canonical_path_and_token_env(monkeypatch: pytest.MonkeyPatch):
    request = MagicMock(
        return_value=_fake_response(200, {"object": "list", "data": []})
    )
    monkeypatch.setattr(management_client.requests, "request", request)
    monkeypatch.setenv("STORAGE_TOKEN", "private-token")

    result = CliRunner().invoke(
        main,
        [
            "storage",
            "vector-stores",
            "--endpoint",
            "http://router.test/api/v1",
            "--token-env",
            "STORAGE_TOKEN",
        ],
    )

    assert result.exit_code == 0, result.output
    assert request.call_args.args[:2] == (
        "GET",
        "http://router.test/api/v1/storage/vector-stores",
    )
    assert (
        request.call_args.kwargs["headers"]["Authorization"] == "Bearer private-token"
    )
    assert "private-token" not in result.output
