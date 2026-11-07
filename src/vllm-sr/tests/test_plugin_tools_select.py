"""Tests for the tools plugin's select block and multi-plugin validation."""

import os
import tempfile

import pytest
import yaml
from cli.config_migration import migrate_config_data
from cli.models import ToolsPluginConfig
from cli.parser import ConfigParseError, parse_user_config
from cli.validator import validate_user_config
from pydantic import ValidationError as PydanticValidationError


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


def _tools_plugin_config(configuration: str) -> str:
    return f"""
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
decisions:
  - name: "test_decision"
    description: "Tools chosen by the decision model"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "test_keywords"
    modelRefs:
      - model: "test_model"
    plugins:
      - type: "tools"
        configuration:
{configuration}
providers:
  models:
    - name: "test_model"
      endpoints:
        - name: "ep1"
          weight: 1
          endpoint: "localhost:8000"
  default_model: "test_model"
"""


class TestToolsPluginSelect:
    """Test the tools plugin's policy and select blocks."""

    def test_tools_select_filter_mode_in_full_config(self):
        """select mode filter with preserve_count and a judge parses end-to-end."""
        temp_path = _write_config(
            _tools_plugin_config(
                """          enabled: true
          policy:
            mode: "filtered"
            block_tools: ["admin.delete"]
          select:
            mode: "filter"
            preserve_count: 3
            on_error: "strip"
            match:
              judge: {preset: "tool_rank", top_k: 3, min_ratio: 0.1}"""
            )
        )

        try:
            config = parse_user_config(temp_path)
            plugin = config.decisions[0].plugins[0]
            assert plugin.type.value == "tools"
            assert plugin.configuration["select"]["mode"] == "filter"
            assert plugin.configuration["select"]["preserve_count"] == 3
            assert plugin.configuration["select"]["match"]["judge"]["top_k"] == 3

            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)

    def test_tools_select_add_mode_in_full_config(self):
        """select mode add with a catalog parses end-to-end."""
        temp_path = _write_config(
            _tools_plugin_config(
                """          enabled: true
          select:
            mode: "add"
            tools_db_path: "config/tools_db.json"
            need_threshold: 0.5
            on_error: "skip\""""
            )
        )

        try:
            config = parse_user_config(temp_path)
            plugin = config.decisions[0].plugins[0]
            assert (
                plugin.configuration["select"]["tools_db_path"]
                == "config/tools_db.json"
            )

            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)

    @pytest.mark.parametrize(
        "select",
        [
            {"mode": "bogus"},
            {"mode": "add"},
            {"mode": "add", "tools_db_path": "tools.json", "preserve_count": 1},
            {"mode": "add", "tools_db_path": "tools.json", "on_error": "keep"},
            {"mode": "filter", "tools_db_path": "tools.json"},
            {"mode": "filter", "on_error": "skip"},
            {"mode": "filter", "preserve_count": 256},
            {"mode": "filter", "need_threshold": 1.5},
        ],
    )
    def test_tools_select_invalid_fields_rejected(self, select):
        with pytest.raises(PydanticValidationError):
            ToolsPluginConfig(enabled=True, select=select)

    def test_tools_select_refuses_a_policy_without_tools(self):
        with pytest.raises(PydanticValidationError):
            ToolsPluginConfig(
                enabled=True, policy={"mode": "none"}, select={"mode": "filter"}
            )

    def test_tools_plugin_fields_outside_policy_rejected_in_full_config(self):
        temp_path = _write_config(
            _tools_plugin_config(
                """          enabled: true
          mode: "none\""""
            )
        )

        try:
            with pytest.raises(
                ConfigParseError, match="removed_field: tools plugin fields mode"
            ):
                parse_user_config(temp_path)
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

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            assert len(config.decisions[0].plugins) == 3

            plugin_types = [p.type.value for p in config.decisions[0].plugins]
            assert "router_replay" in plugin_types
            assert "response_cache" in plugin_types
            assert "system_prompt" in plugin_types

            errors = validate_user_config(config)
            assert len(errors) == 0
        finally:
            os.unlink(temp_path)

    def test_multiple_plugins_with_context(self):
        """Test a context plugin alongside system_prompt, semantic-cache, and router_replay."""
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

        temp_path = _write_config(config_yaml)

        try:
            config = parse_user_config(temp_path)
            assert len(config.decisions[0].plugins) == 4

            plugin_types = [p.type.value for p in config.decisions[0].plugins]
            assert "system_prompt" in plugin_types
            assert "context" in plugin_types
            assert "response_cache" in plugin_types
            assert "router_replay" in plugin_types

            context_plugin = next(
                p for p in config.decisions[0].plugins if p.type.value == "context"
            )
            assert context_plugin.configuration["enabled"] is True
            rag_source = context_plugin.configuration["sources"][0]
            assert rag_source["backend"] == "external_api"
            assert rag_source["match"]["recall"]["top_k"] == 5
            assert (
                rag_source["backend_config"]["endpoint"]
                == "http://rag-service:8000/v1/search"
            )

            # Validate entire config (no errors expected)
            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)
