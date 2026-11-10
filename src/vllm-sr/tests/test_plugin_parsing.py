"""Tests for plugin parsing and validation."""

import pytest
import tempfile
import os
import yaml
from pydantic import ValidationError as PydanticValidationError

from cli.models import (
    PluginConfig,
    PluginType,
    RouterReplayPluginConfig,
    ContextPluginConfig,
    ContextSource,
    ToolsPluginConfig,
)
from cli.parser import ConfigParseError, parse_user_config
from cli.validator import validate_user_config
from recipe_document_test_support import default_recipe_document


def _write_config(config_yaml: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(
            default_recipe_document(yaml.safe_load(config_yaml)), f, sort_keys=False
        )
        return f.name


class TestPluginTypeValidation:
    """Test plugin type validation."""

    def test_valid_plugin_types(self):
        """Test that all valid plugin types are accepted."""
        valid_types = [
            PluginType.RESPONSE_CACHE.value,
            PluginType.SYSTEM_PROMPT.value,
            PluginType.HEADER_MUTATION.value,
            PluginType.ROUTER_REPLAY.value,
            PluginType.CONTEXT.value,
            PluginType.FAST_RESPONSE.value,
            PluginType.REQUEST_PARAMS.value,
            PluginType.RESPONSE_GUARD.value,
            PluginType.TOOLS.value,
            PluginType.PROMPT_CACHE.value,
        ]

        for plugin_type in valid_types:
            plugin = PluginConfig(type=plugin_type, configuration={"enabled": True})
            # plugin.type is now a PluginType enum, compare to enum value
            assert plugin.type.value == plugin_type

    @pytest.mark.parametrize(
        "alias", ["semantic-cache", "semantic_cache", "response-cache"]
    )
    def test_response_cache_aliases_are_no_plugin_types(self, alias):
        with pytest.raises(PydanticValidationError, match="Input should be.*enum"):
            PluginConfig(type=alias, configuration={"enabled": True})

    def test_invalid_plugin_type(self):
        """Test that invalid plugin types are rejected."""
        with pytest.raises(PydanticValidationError, match="Input should be.*enum"):
            PluginConfig(type="invalid_plugin", configuration={"enabled": True})

    def test_new_plugin_types_validate_in_full_config(self):
        """Test newly exposed plugin types validate end-to-end."""
        config_yaml = """
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  keywords:
    - name: "test_keywords"
      operator: "OR"
      keywords: ["test"]
  domains:
    - name: "test"
      description: "Test domain"
decisions:
  - name: "test_decision"
    description: "Decision with extended plugin coverage"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "request_params"
        configuration:
          blocked_params: ["logprobs"]
          max_tokens_limit: 256
          max_n: 1
          strip_unknown: true
      - type: "response_guard"
        configuration:
          rules:
            - when: {type: "keyword", name: "test_keywords"}
              action: "annotate"
              details: true
          on_unknown: "none"
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)


class TestRouterReplayPluginConfig:
    """Test router_replay plugin configuration."""

    def test_valid_router_replay_config(self):
        """Test valid router_replay plugin configuration."""
        config = RouterReplayPluginConfig(
            enabled=True,
            max_records=100,
            capture_request_body=True,
            capture_response_body=False,
            max_body_bytes=2048,
        )
        assert config.enabled is True
        assert config.max_records == 100
        assert config.capture_request_body is True
        assert config.capture_response_body is False
        assert config.max_body_bytes == 2048

    def test_router_replay_config_defaults(self):
        """Test router_replay plugin configuration defaults."""
        config = RouterReplayPluginConfig(enabled=True)
        assert config.enabled is True
        assert config.max_records is None  # Inherits the global capture default
        assert config.capture_request_body is None
        assert config.capture_response_body is None
        assert config.max_body_bytes is None
        assert config.capture_personal_data is None
        assert config.capture_tool_results is None
        assert config.max_tool_trace_bytes is None
        assert config.max_tool_trace_steps is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"enabled": False},
            {"capture_personal_data": False},
            {"capture_tool_results": True},
            {"capture_request_body": False, "capture_response_body": False},
            {"max_tool_trace_bytes": 0, "max_tool_trace_steps": 0},
            {"max_records": 0, "max_body_bytes": 0},
        ],
    )
    def test_router_replay_serialization_preserves_explicit_overrides(self, overrides):
        config = RouterReplayPluginConfig.model_validate(overrides)
        assert config.model_dump(exclude_unset=True) == overrides

    @pytest.mark.parametrize(
        "overrides",
        [
            {"capture_personal_data": "false"},
            {"capture_tool_results": "true"},
            {"capture_request_body": 0},
            {"max_tool_trace_bytes": -1},
            {"max_tool_trace_steps": -1},
            {"max_records": -1},
            {"max_body_bytes": -1},
            {"replay_personal_data": False},
        ],
    )
    def test_router_replay_rejects_invalid_capture_overrides(self, overrides):
        with pytest.raises(PydanticValidationError):
            RouterReplayPluginConfig.model_validate(overrides)

    def test_router_replay_plugin_in_config(self):
        """Test router_replay plugin in full config."""
        config_yaml = """
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  keywords:
    - name: "test_keywords"
      operator: "OR"
      keywords: ["test"]
  domains:
    - name: "test"
      description: "Test domain"
decisions:
  - name: "test_decision"
    description: "Test decision"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "router_replay"
        configuration:
          enabled: true
          max_records: 100
          capture_request_body: true
          capture_response_body: false
          max_body_bytes: 2048
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            assert len(config.decisions) == 1
            assert len(config.decisions[0].plugins) == 1

            plugin = config.decisions[0].plugins[0]
            assert plugin.type.value == "router_replay"
            assert plugin.configuration["enabled"] is True
            assert plugin.configuration["max_records"] == 100
            assert plugin.configuration["capture_request_body"] is True
            assert plugin.configuration["capture_response_body"] is False
            assert plugin.configuration["max_body_bytes"] == 2048

            # Validate the config
            errors = validate_user_config(config)
            assert len(errors) == 0
        finally:
            os.unlink(temp_path)


class TestToolsPluginConfig:
    """Test the tools plugin schema and its privacy-policy round trip."""

    def test_strip_history_round_trip(self):
        privacy = ToolsPluginConfig.model_validate(
            {
                "enabled": True,
                "policy": {"mode": "none", "strip_tool_history": True},
            }
        )
        dumped = privacy.model_dump(exclude_none=True)
        assert dumped["policy"]["strip_tool_history"] is True

    def test_strip_history_requires_mode_none(self):
        with pytest.raises(
            PydanticValidationError, match="strip_tool_history requires mode=none"
        ):
            ToolsPluginConfig.model_validate(
                {
                    "enabled": True,
                    "policy": {"mode": "passthrough", "strip_tool_history": True},
                }
            )

    def test_policy_fields_at_the_top_are_rejected(self):
        with pytest.raises(PydanticValidationError):
            ToolsPluginConfig.model_validate({"enabled": True, "mode": "none"})

    def test_enabled_tools_plugin_rejects_unknown_mode(self):
        with pytest.raises(PydanticValidationError):
            ToolsPluginConfig.model_validate(
                {"enabled": True, "policy": {"mode": "future_mode"}}
            )


class TestPluginConfigurationValidation:
    """Test plugin configuration validation."""

    def test_invalid_router_replay_config(self):
        """Test that invalid router_replay configuration is caught."""
        config_yaml = """
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  keywords:
    - name: "test_keywords"
      operator: "OR"
      keywords: ["test"]
  domains:
    - name: "test"
      description: "Test domain"
decisions:
  - name: "test_decision"
    description: "Test decision"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "router_replay"
        configuration:
          enabled: true
          max_records: "invalid"  # Should be int
          capture_request_body: "yes"  # Should be bool
"""

        temp_path = _write_config(config_yaml)

        try:
            with pytest.raises(ConfigParseError) as exc_info:
                parse_user_config(temp_path)
            message = str(exc_info.value)
            assert "configuration.max_records" in message
            assert "configuration.capture_request_body" in message
        finally:
            os.unlink(temp_path)

    def test_invalid_response_cache_config(self):
        """Test that invalid response_cache configuration is caught."""
        config_yaml = """
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  keywords:
    - name: "test_keywords"
      operator: "OR"
      keywords: ["test"]
  domains:
    - name: "test"
      description: "Test domain"
decisions:
  - name: "test_decision"
    description: "Test decision"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "response_cache"
        configuration:
          enabled: "yes"  # Should be bool
          ttl_seconds: -1  # Should be >= 0
"""

        temp_path = _write_config(config_yaml)

        try:
            with pytest.raises(ConfigParseError) as exc_info:
                parse_user_config(temp_path)
            assert "configuration.enabled" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_missing_required_fields(self):
        """Test that missing required fields are caught."""
        config_yaml = """
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
decisions:
  - name: "test_decision"
    description: "Test decision"
    priority: 100
    rules:
      operator: "OR"
      conditions: []
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "response_cache"
        configuration:
          # Missing required 'enabled' field
          ttl_seconds: 60
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            # SemanticCachePluginConfig requires 'enabled' field, so validation should fail
            assert isinstance(errors, list)
            missing_field_message = (
                "Expected validation errors for missing required field"
            )
            assert len(errors) > 0, missing_field_message
            # Check that the error mentions the missing field
            error_messages = [str(e) for e in errors]
            enabled_field_message = (
                f"Expected error about missing 'enabled' field, got: {error_messages}"
            )
            enabled_field_reported = any(
                "enabled" in msg.lower() for msg in error_messages
            )
            assert enabled_field_reported, enabled_field_message
        finally:
            os.unlink(temp_path)


class TestContextPluginConfig:
    """Test context plugin configuration."""

    def test_valid_rag_source_all_fields(self):
        config = ContextPluginConfig(
            enabled=True,
            sources=[
                {
                    "type": "rag",
                    "backend": "external_api",
                    "match": {
                        "recall": {
                            "method": "embedding",
                            "top_k": 5,
                            "min_similarity": 0.75,
                        },
                        "judge": {"preset": "relevance", "top_k": 2},
                    },
                    "max_context_length": 4096,
                    "backend_config": {
                        "endpoint": "http://rag-service:8000/v1/search",
                        "request_format": "openai",
                        "timeout_seconds": 10,
                        "max_response_bytes": 4194304,
                    },
                    "on_failure": "skip",
                    "cache_results": True,
                    "cache_ttl_seconds": 300,
                }
            ],
            injection={"mode": "tool_role"},
        )
        source = config.sources[0]
        assert source.type == "rag"
        assert source.backend == "external_api"
        assert source.match["judge"]["top_k"] == 2
        assert source.backend_config["max_response_bytes"] == 4194304
        assert source.on_failure == "skip"
        assert source.cache_ttl_seconds == 300
        assert config.injection.mode == "tool_role"

    def test_openai_response_limit_is_preserved(self):
        source = ContextSource(
            type="rag",
            backend="openai",
            backend_config={
                "vector_store_id": "vs_123",
                "api_key": "secret",
                "max_response_bytes": 4194304,
            },
        )
        assert source.backend_config["max_response_bytes"] == 4194304

    def test_rag_source_required_fields_only(self):
        source = ContextSource(type="rag", backend="milvus")
        assert source.match is None
        assert source.max_context_length is None
        assert source.backend_config is None
        assert source.on_failure is None
        assert source.cache_ttl_seconds is None

    def test_source_takes_only_its_own_fields(self):
        with pytest.raises(
            PydanticValidationError, match="a rag source needs a backend"
        ):
            ContextSource(type="rag")
        with pytest.raises(
            PydanticValidationError, match="a memory source takes no backend"
        ):
            ContextSource(type="memory", backend="milvus")
        with pytest.raises(
            PydanticValidationError, match="a rag source takes no store"
        ):
            ContextSource(type="rag", backend="milvus", store={"auto": True})
        assert ContextSource(type="memory", store={"auto": False}).store.auto is False

    def test_field_constraints(self):
        with pytest.raises(PydanticValidationError):
            ContextSource(type="rag", backend="milvus", cache_ttl_seconds=0)
        with pytest.raises(PydanticValidationError):
            ContextSource(type="rag", backend="milvus", max_context_length=-1)
        with pytest.raises(PydanticValidationError):
            ContextSource(type="rag", backend="pinecone")
        with pytest.raises(PydanticValidationError, match="retrieves sources"):
            ContextPluginConfig(enabled=True)
        assert ContextPluginConfig(enabled=False).sources is None

    def test_context_plugin_in_full_config(self):
        config_yaml = """
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
signals:
  keywords:
    - name: "test_keywords"
      operator: "OR"
      keywords: ["test"]
  domains:
    - name: "test"
      description: "Test domain"
decisions:
  - name: "test_decision"
    description: "Test decision with context"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "context"
        configuration:
          enabled: true
          sources:
            - type: "rag"
              backend: "external_api"
              match:
                recall: {method: embedding, top_k: 5, min_similarity: 0.75}
              on_failure: "skip"
              cache_results: true
              cache_ttl_seconds: 300
              backend_config:
                endpoint: "http://rag-service:8000/v1/search"
                request_format: "openai"
                timeout_seconds: 10
          injection:
            mode: "tool_role"
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            plugin = config.decisions[0].plugins[0]
            assert plugin.type.value == "context"
            source = plugin.configuration["sources"][0]
            assert source["backend"] == "external_api"
            assert source["match"]["recall"]["top_k"] == 5
            assert source["backend_config"]["request_format"] == "openai"

            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)

    def test_invalid_context_config_in_full_yaml(self):
        config_yaml = """
listeners:
  - name: "http-8888"
    address: "0.0.0.0"
    port: 8888
decisions:
  - name: "test_decision"
    description: "Test decision with an invalid context source"
    priority: 100
    rules:
      operator: "OR"
      conditions: []
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "context"
        configuration:
          enabled: true
          sources:
            - type: "rag"
              backend: "external_api"
              cache_ttl_seconds: "invalid"
"""

        temp_path = _write_config(config_yaml)

        try:
            with pytest.raises(ConfigParseError) as exc_info:
                parse_user_config(temp_path)
            assert "cache_ttl_seconds" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
