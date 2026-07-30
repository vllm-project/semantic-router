"""Tests for external_api request-format validation in hybrid RAG configs.

Covers the FAUST-BENCHOU feedback on PR #2507: the Python CLI must validate
nested hybrid ``external_api`` child configs through the same contract as a
top-level ``external_api`` backend, mirroring the Go-side
``validateHybridRAGChildBackend``.
"""

import pytest
from pydantic import ValidationError

from cli.models import RAGPluginConfig


def _external_api_cfg(request_format: str = "pinecone") -> dict:
    return {"request_format": request_format}


class TestTopLevelExternalApi:
    """Baseline: top-level external_api validation (unchanged behaviour)."""

    def test_accepts_valid_format(self):
        RAGPluginConfig(
            enabled=True,
            backend="external_api",
            backend_config=_external_api_cfg("weaviate"),
        )

    def test_rejects_invalid_format(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="external_api",
                backend_config={"request_format": "unsupported"},
            )

    def test_rejects_missing_format(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="external_api",
                backend_config={},
            )


class TestHybridPrimaryExternalApi:
    """Hybrid config whose primary backend is external_api."""

    def test_accepts_valid_primary(self):
        RAGPluginConfig(
            enabled=True,
            backend="hybrid",
            backend_config={
                "primary": "external_api",
                "primary_config": _external_api_cfg("elasticsearch"),
                "fallback": "milvus",
                "fallback_config": {"collection": "docs"},
            },
        )

    def test_rejects_invalid_primary_format(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="hybrid",
                backend_config={
                    "primary": "external_api",
                    "primary_config": {"request_format": "bogus"},
                    "fallback": "milvus",
                    "fallback_config": {"collection": "docs"},
                },
            )

    def test_rejects_missing_primary_config(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="hybrid",
                backend_config={
                    "primary": "external_api",
                    "fallback": "milvus",
                    "fallback_config": {"collection": "docs"},
                },
            )


class TestHybridFallbackExternalApi:
    """Hybrid config whose fallback backend is external_api."""

    def test_accepts_valid_fallback(self):
        RAGPluginConfig(
            enabled=True,
            backend="hybrid",
            backend_config={
                "primary": "milvus",
                "primary_config": {"collection": "docs"},
                "fallback": "external_api",
                "fallback_config": _external_api_cfg("custom"),
            },
        )

    def test_rejects_invalid_fallback_format(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="hybrid",
                backend_config={
                    "primary": "milvus",
                    "primary_config": {"collection": "docs"},
                    "fallback": "external_api",
                    "fallback_config": {"request_format": "nope"},
                },
            )

    def test_rejects_missing_fallback_config(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="hybrid",
                backend_config={
                    "primary": "milvus",
                    "primary_config": {"collection": "docs"},
                    "fallback": "external_api",
                },
            )


class TestHybridBothExternalApi:
    """Hybrid config where both primary and fallback are external_api."""

    def test_accepts_both_valid(self):
        RAGPluginConfig(
            enabled=True,
            backend="hybrid",
            backend_config={
                "primary": "external_api",
                "primary_config": _external_api_cfg("pinecone"),
                "fallback": "external_api",
                "fallback_config": _external_api_cfg("weaviate"),
            },
        )

    def test_rejects_when_primary_invalid_fallback_valid(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="hybrid",
                backend_config={
                    "primary": "external_api",
                    "primary_config": {"request_format": "bad"},
                    "fallback": "external_api",
                    "fallback_config": _external_api_cfg("pinecone"),
                },
            )

    def test_rejects_when_primary_valid_fallback_invalid(self):
        with pytest.raises(ValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="hybrid",
                backend_config={
                    "primary": "external_api",
                    "primary_config": _external_api_cfg("pinecone"),
                    "fallback": "external_api",
                    "fallback_config": {"request_format": "bad"},
                },
            )


class TestHybridNonExternalApi:
    """Hybrid configs with no external_api children should not be affected."""

    def test_accepts_milvus_only(self):
        RAGPluginConfig(
            enabled=True,
            backend="hybrid",
            backend_config={
                "primary": "milvus",
                "primary_config": {"collection": "docs"},
            },
        )

    def test_accepts_milvus_openai(self):
        RAGPluginConfig(
            enabled=True,
            backend="hybrid",
            backend_config={
                "primary": "milvus",
                "primary_config": {"collection": "docs"},
                "fallback": "openai",
                "fallback_config": {
                    "vector_store_id": "vs_abc",
                    "api_key": "sk-test",
                },
            },
        )
