"""Tests for `vllm-sr rag list` (#2117).

The command lists vector stores from the router's ``GET /v1/vector_stores``
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
import requests
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

main = importlib.import_module("cli.main").main
rag_command = importlib.import_module("cli.commands.rag")


def _fake_response(status_code: int, json_body=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
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


def test_normalize_endpoint_default_uses_api_port():
    assert rag_command._normalize_endpoint("").endswith(":8080/v1/vector_stores")


def test_normalize_endpoint_accepts_base_url():
    assert (
        rag_command._normalize_endpoint("http://localhost:8180")
        == "http://localhost:8180/v1/vector_stores"
    )


def test_normalize_endpoint_accepts_full_path():
    full = "http://localhost:8080/v1/vector_stores"
    assert rag_command._normalize_endpoint(full) == full


def test_rag_list_registered_on_top_level_help():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "rag" in result.output


def test_rag_list_help_describes_subcommand():
    runner = CliRunner()

    result = runner.invoke(main, ["rag", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output


def test_rag_list_prints_vector_stores(caplog, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        rag_command.requests, "get", lambda *a, **k: _fake_response(200, _TWO_STORES)
    )

    with caplog.at_level("INFO"):
        result = runner.invoke(main, ["rag", "list"])

    assert result.exit_code == 0
    combined = "\n".join(record.message for record in caplog.records)

    assert "Vector stores (2)" in combined
    assert "docs-prod" in combined
    assert "vs_abc123" in combined
    assert "support-kb" in combined
    assert "milvus" in combined
    assert "10 total" in combined
    assert "1 failed" in combined


def test_rag_list_reports_no_vector_stores(caplog, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        rag_command.requests,
        "get",
        lambda *a, **k: _fake_response(200, {"object": "list", "data": []}),
    )

    with caplog.at_level("INFO"):
        result = runner.invoke(main, ["rag", "list"])

    assert result.exit_code == 0
    combined = "\n".join(record.message for record in caplog.records)
    assert "Vector stores (0)" in combined
    assert "(none created)" in combined


def test_rag_list_reports_feature_disabled(caplog, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    monkeypatch.setattr(
        rag_command.requests,
        "get",
        lambda *a, **k: _fake_response(503, text="disabled"),
    )

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["rag", "list"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "not enabled" in combined


def test_rag_list_reports_unreachable_router(caplog, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()

    def _boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(rag_command.requests, "get", _boom)

    with caplog.at_level("ERROR"):
        result = runner.invoke(main, ["rag", "list"])

    assert result.exit_code == 1
    combined = "\n".join(record.message for record in caplog.records)
    assert "Router is not running" in combined
