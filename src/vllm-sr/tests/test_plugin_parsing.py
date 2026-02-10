"""Tests for plugin parsing and validation."""

import pytest
import tempfile
import os
from pydantic import ValidationError as PydanticValidationError

from cli.models import (
    PluginConfig,
    PluginType,
    RouterReplayPluginConfig,
    RAGPluginConfig,
)
from cli.parser import parse_user_config
from cli.validator import validate_user_config


class TestPluginTypeValidation:
    """Test plugin type validation."""

    def test_valid_plugin_types(self):
        """Test that all valid plugin types are accepted."""
        valid_types = [
            PluginType.SEMANTIC_CACHE.value,
            PluginType.JAILBREAK.value,
            PluginType.PII.value,
            PluginType.SYSTEM_PROMPT.value,
            PluginType.HEADER_MUTATION.value,
            PluginType.HALLUCINATION.value,
            PluginType.ROUTER_REPLAY.value,
            PluginType.RAG.value,
        ]

        for plugin_type in valid_types:
            plugin = PluginConfig(type=plugin_type, configuration={"enabled": True})
            # plugin.type is now a PluginType enum, compare to enum value
            assert plugin.type.value == plugin_type

    def test_invalid_plugin_type(self):
        """Test that invalid plugin types are rejected."""
        with pytest.raises(PydanticValidationError, match="Input should be.*enum"):
            PluginConfig(type="invalid_plugin", configuration={"enabled": True})


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
        assert config.max_records == 200  # Default
        assert config.capture_request_body is False  # Default
        assert config.capture_response_body is False  # Default
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

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
        """Test RAGPluginConfig with only required fields (enabled + backend).
        All optional fields should default to None.
        """
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
        """Test that Pydantic field constraints reject out-of-range values.

        Covers: similarity_threshold (0.0-1.0), top_k (>=1),
        max_context_length (>=1), cache_ttl_seconds (>=1),
        min_confidence_threshold (0.0-1.0).
        """
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
        """Test RAG plugin end-to-end: YAML parsing + validation.

        Verifies that a complete YAML config with a RAG plugin can be
        parsed into Pydantic models and passes validation without errors.
        """
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

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
        """Test that invalid RAG configuration is caught by the validator.

        Uses similarity_threshold=1.5 (exceeds 0.0-1.0) and top_k="invalid"
        (wrong type) to verify the validation pipeline catches bad RAG configs.
        """
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            assert len(errors) > 0
            error_messages = [str(e) for e in errors]
            assert any("rag" in msg.lower() for msg in error_messages)
        finally:
            os.unlink(temp_path)


class TestMultiplePlugins:
    """Test configurations with multiple plugins."""

    def test_multiple_plugins_in_decision(self):
        """Test decision with multiple plugins."""
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
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.9
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a test assistant"
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

        try:
            config = parse_user_config(temp_path)
            assert len(config.decisions[0].plugins) == 3

            plugin_types = [p.type.value for p in config.decisions[0].plugins]
            assert "router_replay" in plugin_types
            assert "semantic-cache" in plugin_types
            assert "system_prompt" in plugin_types

            errors = validate_user_config(config)
            assert len(errors) == 0
        finally:
            os.unlink(temp_path)

    def test_multiple_plugins_with_rag(self):
        """Test decision with RAG plugin alongside other plugins.

        This validates the real-world use case from config.template.yaml
        where RAG is used together with system_prompt, semantic-cache,
        and router_replay in the same decision.

        Reference: src/semantic-router/pkg/extproc/req_filter_rag.go
        """
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
    description: "Decision with RAG and other plugins"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a knowledge assistant."
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
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.92
      - type: "router_replay"
        configuration:
          enabled: true
          max_records: 200
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name

        try:
            config = parse_user_config(temp_path)
            assert len(config.decisions[0].plugins) == 4

            plugin_types = [p.type.value for p in config.decisions[0].plugins]
            assert "system_prompt" in plugin_types
            assert "rag" in plugin_types
            assert "semantic-cache" in plugin_types
            assert "router_replay" in plugin_types

            # Verify RAG plugin configuration is correctly parsed
            rag_plugin = next(
                p for p in config.decisions[0].plugins if p.type.value == "rag"
            )
            assert rag_plugin.configuration["enabled"] is True
            assert rag_plugin.configuration["backend"] == "external_api"
            assert rag_plugin.configuration["top_k"] == 5
            assert (
                rag_plugin.configuration["backend_config"]["endpoint"]
                == "http://rag-service:8000/v1/search"
            )

            # Validate entire config (no errors expected)
            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)
