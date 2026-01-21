"""Tests for plugin parsing and validation."""

import pytest
import tempfile
import os
from pydantic import ValidationError as PydanticValidationError

from cli.models import (
    PluginConfig,
    PluginType,
    RouterReplayPluginConfig,
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
