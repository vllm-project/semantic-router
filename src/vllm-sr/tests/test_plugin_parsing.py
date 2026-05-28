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
    RAGPluginConfig,
    ToolSelectionPluginConfig,
)
from cli.config_migration import migrate_config_data
from cli.parser import parse_user_config
from cli.validator import validate_user_config


def _write_config(config_yaml: str) -> str:
    data = yaml.safe_load(config_yaml)
    providers = data.get("providers") if isinstance(data, dict) else None
    models = providers.get("models", []) if isinstance(providers, dict) else []
    if isinstance(data, dict) and (
        "signals" in data
        or "decisions" in data
        or any(
            isinstance(model, dict)
            and any(
                key in model
                for key in (
                    "endpoints",
                    "access_key",
                    "reasoning_family",
                    "param_size",
                    "context_window_size",
                    "description",
                    "capabilities",
                    "quality_score",
                    "modality",
                    "tags",
                )
            )
            for model in models
        )
    ):
        data = migrate_config_data(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(data, f, sort_keys=False)
        return f.name


class TestPluginTypeValidation:
    """Test plugin type validation."""

    def test_valid_plugin_types(self):
        """Test that all valid plugin types are accepted."""
        valid_types = [
            PluginType.SEMANTIC_CACHE.value,
            PluginType.SYSTEM_PROMPT.value,
            PluginType.HEADER_MUTATION.value,
            PluginType.HALLUCINATION.value,
            PluginType.ROUTER_REPLAY.value,
            PluginType.MEMORY.value,
            PluginType.RAG.value,
            PluginType.IMAGE_GEN.value,
            PluginType.FAST_RESPONSE.value,
            PluginType.REQUEST_PARAMS.value,
            PluginType.RESPONSE_JAILBREAK.value,
            PluginType.TOOLS.value,
            PluginType.TOOL_SELECTION.value,
        ]

        for plugin_type in valid_types:
            plugin = PluginConfig(type=plugin_type, configuration={"enabled": True})
            # plugin.type is now a PluginType enum, compare to enum value
            assert plugin.type.value == plugin_type

    def test_invalid_plugin_type(self):
        """Test that invalid plugin types are rejected."""
        with pytest.raises(PydanticValidationError, match="Input should be.*enum"):
            PluginConfig(type="invalid_plugin", configuration={"enabled": True})

    def test_new_plugin_types_validate_in_full_config(self):
        """Test newly exposed plugin types validate end-to-end."""
        config_yaml = """
version: v0.1
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
      - type: "response_jailbreak"
        configuration:
          enabled: true
          threshold: 0.7
          action: "header"
      - type: "image_gen"
        configuration:
          enabled: true
          backend: "openai"
          backend_config:
            api_key: "test-key"
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
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
        assert config.max_records == 10000  # Default
        assert config.capture_request_body is True  # Default
        assert config.capture_response_body is True  # Default
        assert config.max_body_bytes == 4096  # Default

    def test_router_replay_plugin_in_config(self):
        """Test router_replay plugin in full config."""
        config_yaml = """
version: v0.1
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
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
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


class TestPluginConfigurationValidation:
    """Test plugin configuration validation."""

    def test_invalid_router_replay_config(self):
        """Test that invalid router_replay configuration is caught."""
        config_yaml = """
version: v0.1
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
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            assert len(errors) > 0
            # Check that error mentions router_replay
            error_messages = [str(e) for e in errors]
            assert any("router_replay" in msg.lower() for msg in error_messages)
        finally:
            os.unlink(temp_path)

    def test_invalid_semantic_cache_config(self):
        """Test that invalid semantic-cache configuration is caught."""
        config_yaml = """
version: v0.1
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
      - type: "semantic-cache"
        configuration:
          enabled: "yes"  # Should be bool
          similarity_threshold: 1.5  # Should be 0.0-1.0
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            assert len(errors) > 0
            # Check that error mentions semantic-cache
            error_messages = [str(e) for e in errors]
            assert any("semantic-cache" in msg.lower() for msg in error_messages)
        finally:
            os.unlink(temp_path)

    def test_missing_required_fields(self):
        """Test that missing required fields are caught."""
        config_yaml = """
version: v0.1
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
      - type: "semantic-cache"
        configuration:
          # Missing required 'enabled' field
          similarity_threshold: 0.8
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            # SemanticCachePluginConfig requires 'enabled' field, so validation should fail
            assert isinstance(errors, list)
            assert (
                len(errors) > 0
            ), "Expected validation errors for missing required field"
            # Check that the error mentions the missing field
            error_messages = [str(e) for e in errors]
            assert any(
                "enabled" in msg.lower() for msg in error_messages
            ), f"Expected error about missing 'enabled' field, got: {error_messages}"
        finally:
            os.unlink(temp_path)


class TestRAGPluginConfig:
    """Test RAG plugin configuration."""

    def test_valid_rag_config_all_fields(self):
        """Test RAGPluginConfig accepts all valid fields."""
        config = RAGPluginConfig(
            enabled=True,
            backend="external_api",
            similarity_threshold=0.75,
            top_k=5,
            max_context_length=4096,
            injection_mode="tool_role",
            backend_config={
                "endpoint": "http://rag-service:8000/v1/search",
                "request_format": "openai",
                "timeout_seconds": 10,
            },
            on_failure="skip",
            cache_results=True,
            cache_ttl_seconds=300,
            min_confidence_threshold=0.5,
        )
        assert config.enabled is True
        assert config.backend == "external_api"
        assert config.similarity_threshold == 0.75
        assert config.top_k == 5
        assert config.max_context_length == 4096
        assert config.injection_mode == "tool_role"
        assert config.backend_config["endpoint"] == "http://rag-service:8000/v1/search"
        assert config.backend_config["request_format"] == "openai"
        assert config.on_failure == "skip"
        assert config.cache_results is True
        assert config.cache_ttl_seconds == 300
        assert config.min_confidence_threshold == 0.5

    def test_rag_config_required_fields_only(self):
        """Test RAGPluginConfig with only required fields; optional fields default to None."""
        config = RAGPluginConfig(enabled=True, backend="milvus")
        assert config.enabled is True
        assert config.backend == "milvus"
        assert config.similarity_threshold is None
        assert config.top_k is None
        assert config.max_context_length is None
        assert config.injection_mode is None
        assert config.backend_config is None
        assert config.on_failure is None
        assert config.cache_results is None
        assert config.cache_ttl_seconds is None
        assert config.min_confidence_threshold is None

    def test_rag_config_missing_required_fields(self):
        """Test that missing required fields (enabled, backend) raise errors."""
        with pytest.raises(PydanticValidationError, match="enabled"):
            RAGPluginConfig(backend="external_api")

        with pytest.raises(PydanticValidationError, match="backend"):
            RAGPluginConfig(enabled=True)

    def test_rag_config_field_constraints(self):
        """Test that Pydantic rejects out-of-range RAG field values."""
        # similarity_threshold > 1.0
        with pytest.raises(PydanticValidationError):
            RAGPluginConfig(enabled=True, backend="milvus", similarity_threshold=1.1)

        # similarity_threshold < 0.0
        with pytest.raises(PydanticValidationError):
            RAGPluginConfig(enabled=True, backend="milvus", similarity_threshold=-0.1)

        # top_k must be >= 1
        with pytest.raises(PydanticValidationError):
            RAGPluginConfig(enabled=True, backend="milvus", top_k=0)

        # cache_ttl_seconds must be >= 1
        with pytest.raises(PydanticValidationError):
            RAGPluginConfig(enabled=True, backend="milvus", cache_ttl_seconds=0)

        # min_confidence_threshold > 1.0
        with pytest.raises(PydanticValidationError):
            RAGPluginConfig(
                enabled=True, backend="milvus", min_confidence_threshold=1.1
            )

    def test_rag_plugin_in_full_config(self):
        """Test end-to-end RAG plugin YAML parsing and validation."""
        config_yaml = """
version: v0.1
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
    description: "Test decision with RAG"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "rag"
        configuration:
          enabled: true
          backend: "external_api"
          top_k: 5
          similarity_threshold: 0.75
          injection_mode: "tool_role"
          on_failure: "skip"
          cache_results: true
          cache_ttl_seconds: 300
          backend_config:
            endpoint: "http://rag-service:8000/v1/search"
            request_format: "openai"
            timeout_seconds: 10
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            assert len(config.decisions) == 1
            assert len(config.decisions[0].plugins) == 1

            plugin = config.decisions[0].plugins[0]
            assert plugin.type.value == "rag"
            assert plugin.configuration["enabled"] is True
            assert plugin.configuration["backend"] == "external_api"
            assert plugin.configuration["top_k"] == 5
            assert plugin.configuration["similarity_threshold"] == 0.75
            assert plugin.configuration["injection_mode"] == "tool_role"
            assert plugin.configuration["on_failure"] == "skip"
            assert plugin.configuration["cache_results"] is True
            assert plugin.configuration["cache_ttl_seconds"] == 300
            assert (
                plugin.configuration["backend_config"]["endpoint"]
                == "http://rag-service:8000/v1/search"
            )
            assert plugin.configuration["backend_config"]["request_format"] == "openai"

            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)

    def test_invalid_rag_config_in_full_yaml(self):
        """Test that invalid RAG YAML is caught by the validator."""
        config_yaml = """
version: v0.1
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
    description: "Test decision with invalid RAG"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "rag"
        configuration:
          enabled: true
          backend: "external_api"
          similarity_threshold: 1.5
          top_k: "invalid"
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            assert len(errors) > 0
            error_messages = [str(e) for e in errors]
            assert any("rag" in msg.lower() for msg in error_messages)
        finally:
            os.unlink(temp_path)


class TestToolSelectionAdvancedFiltering:
    """Test tool_selection plugin's advanced_filtering subtree.

    Pins down the schema parity with the Go-side
    `config.AdvancedToolFilteringConfig`. Before this surface was modeled,
    `ToolSelectionPluginConfig` had no `advanced_filtering` field and Pydantic
    silently dropped the subtree, so users opting into
    `retrieval_strategy: hybrid_history` ended up with a config equivalent to
    no advanced filtering at all.
    """

    def test_hybrid_history_round_trips_without_field_loss(self):
        """advanced_filtering.hybrid_history survives model_validate -> model_dump."""
        cfg = ToolSelectionPluginConfig.model_validate(
            {
                "enabled": True,
                "mode": "add",
                "tools_db_path": "config/tools_db.json",
                "top_k": 5,
                "advanced_filtering": {
                    "enabled": True,
                    "retrieval_strategy": "hybrid_history",
                    "hybrid_history": {
                        "history_horizon": 8,
                        "weight_history_transition": 2.0,
                    },
                },
            }
        )

        dump = cfg.model_dump(exclude_none=True)
        assert "advanced_filtering" in dump
        assert dump["advanced_filtering"]["retrieval_strategy"] == "hybrid_history"
        assert dump["advanced_filtering"]["hybrid_history"]["history_horizon"] == 8
        assert (
            dump["advanced_filtering"]["hybrid_history"]["weight_history_transition"]
            == 2.0
        )

    def test_invalid_retrieval_strategy_rejected(self):
        """retrieval_strategy must be 'weighted' or 'hybrid_history'."""
        with pytest.raises(PydanticValidationError):
            ToolSelectionPluginConfig.model_validate(
                {
                    "enabled": True,
                    "advanced_filtering": {
                        "enabled": True,
                        "retrieval_strategy": "bogus",
                    },
                }
            )

    def test_invalid_threshold_rejected(self):
        """category_confidence_threshold must be in [0.0, 1.0]."""
        with pytest.raises(PydanticValidationError):
            ToolSelectionPluginConfig.model_validate(
                {
                    "enabled": True,
                    "advanced_filtering": {
                        "enabled": True,
                        "category_confidence_threshold": 1.5,
                    },
                }
            )
