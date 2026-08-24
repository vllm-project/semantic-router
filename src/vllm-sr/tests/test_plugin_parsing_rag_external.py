"""External-API RAG backend parsing and validation.

Split out of test_plugin_parsing.py, which reached the repository's 800-line
per-file lint limit. Sibling of test_rag_external_api_hybrid_validator.py.
"""

import pytest

from cli.parser import parse_user_config
from cli.validator import validate_user_config

from .test_plugin_parsing import _write_config


class TestExternalAPIRAGBackend:
    """Typed request-format contracts for `backend: external_api`."""

    def test_external_api_accepts_router_supported_request_formats(
        self, request_format
    ):
        """Python validation accepts every request format supported by the router."""
        backend_config = {
            "endpoint": "http://rag-service:8000/v1/search",
            "request_format": request_format,
            "provider_options": {"namespace": "docs", "preserve_me": True},
        }

        config = RAGPluginConfig(
            enabled=True,
            backend="external_api",
            backend_config=backend_config,
        )

        assert config.backend_config == backend_config
        assert config.model_dump()["backend_config"] == backend_config

    @pytest.mark.parametrize(
        "backend_config",
        [
            None,
            {},
            {"request_format": ""},
            {"request_format": "openai"},
            {"request_format": "elastic_search"},
        ],
    )
    def test_external_api_rejects_missing_or_unsupported_request_format(
        self, backend_config
    ):
        """Python validation rejects formats that the router loader rejects."""
        with pytest.raises(PydanticValidationError, match="request_format"):
            RAGPluginConfig(
                enabled=True,
                backend="external_api",
                backend_config=backend_config,
            )

    def test_openai_file_search_backend_remains_distinct(self):
        """The OpenAI file-search backend is not an external API format."""
        config = RAGPluginConfig(
            enabled=True,
            backend="openai",
            backend_config={"vector_store_id": "vs_test", "api_key": "test-key"},
        )
        assert config.backend_config["vector_store_id"] == "vs_test"

    def test_external_api_full_yaml_rejects_openai_and_missing_request_format(self):
        """Full YAML validation rejects OpenAI format and an omitted format."""
        valid_config_yaml = """
version: v0.1
listeners:
  - {name: "http-8888", address: "0.0.0.0", port: 8888}
signals:
  keywords:
    - {name: "test_keywords", operator: "OR", keywords: ["test"]}
decisions:
  - name: "test_decision"
    description: "Test external API request format validation"
    priority: 100
    rules: {operator: "OR", conditions: [{type: "keyword", name: "test_keywords"}]}
    modelRefs: [{model: "test_model"}]
    plugins:
      - type: "rag"
        configuration:
          enabled: true
          backend: "external_api"
          backend_config:
            endpoint: "http://rag-service:8000/v1/search"
            request_format: "pinecone"
providers:
  models:
    - name: "test_model"
      endpoints:
        - {name: "ep1", weight: 1, endpoint: "localhost:8000"}
  default_model: "test_model"
"""
        invalid_configs = {
            "openai": valid_config_yaml.replace(
                'request_format: "pinecone"', 'request_format: "openai"'
            ),
            "missing": valid_config_yaml.replace(
                '            request_format: "pinecone"\n', ""
            ),
        }

        for case, config_yaml in invalid_configs.items():
            temp_path = _write_config(config_yaml)
            try:
                config = parse_user_config(temp_path)
                errors = validate_user_config(config)
                error_messages = [str(error) for error in errors]
                assert any(
                    "rag" in message.lower() and "request_format" in message
                    for message in error_messages
                ), f"{case} request_format unexpectedly passed: {error_messages}"
            finally:
                os.unlink(temp_path)
