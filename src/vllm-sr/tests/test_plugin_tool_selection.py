"""Tests for tool_selection plugin parsing and multi-plugin validation."""

import os
import tempfile

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from cli.models import ToolSelectionPluginConfig
from cli.parser import parse_user_config
from cli.validator import validate_user_config


def _write_config(plugins: list[dict]) -> str:
    data = {
        "version": "v0.4",
        "listeners": [{"name": "http-8888", "address": "0.0.0.0", "port": 8888}],
        "models": [
            {
                "name": "test_model",
                "card": {
                    "description": "Test model",
                    "capabilities": ["chat"],
                },
                "connections": [
                    {
                        "provider": "vllm",
                        "endpoint": "http://localhost:8000/v1",
                        "model": "test_model",
                    }
                ],
            }
        ],
        "recipes": [
            {
                "name": "test_recipe",
                "document": {
                    "signals": {
                        "keywords": [
                            {
                                "name": "test_keywords",
                                "operator": "OR",
                                "keywords": ["test"],
                            }
                        ],
                        "domains": [{"name": "test", "description": "Test domain"}],
                    },
                    "decisions": [
                        {
                            "name": "test_decision",
                            "description": "Test decision",
                            "priority": 100,
                            "rules": {
                                "operator": "OR",
                                "conditions": [
                                    {"type": "keyword", "name": "test_keywords"}
                                ],
                            },
                            "plugins": plugins,
                        }
                    ],
                },
            }
        ],
        "entrypoints": [
            {
                "name": "test_entrypoint",
                "recipe": "test_recipe",
                "assignments": {"test_decision": {"models": [{"model": "test_model"}]}},
            }
        ],
        "global": {
            "services": {
                "backend_dispatch": {
                    "bind_address": "127.0.0.1",
                    "port": 8180,
                    "audience": "vllm-sr.backend-dispatch",
                    "capability_ttl": "30s",
                    "max_request_body_bytes": 64 << 20,
                },
                "backend_egress": {
                    "policy_file": "/app/config/backend-egress-policy.yaml"
                },
            }
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(data, f, sort_keys=False)
        return f.name


class TestToolSelectionPluginConfig:
    """Test tool_selection plugin configuration."""

    def test_tool_selection_filter_mode_in_full_config(self):
        """mode: filter with relevance_threshold + preserve_count parses end-to-end."""
        temp_path = _write_config(
            [
                {
                    "type": "tool_selection",
                    "configuration": {
                        "enabled": True,
                        "mode": "filter",
                        "relevance_threshold": 0.4,
                        "preserve_count": 3,
                    },
                }
            ]
        )

        try:
            config = parse_user_config(temp_path)
            plugin = config.recipes[0].document.decisions[0].plugins[0]
            assert plugin.type.value == "tool_selection"
            assert plugin.configuration["mode"] == "filter"
            assert plugin.configuration["relevance_threshold"] == 0.4
            assert plugin.configuration["preserve_count"] == 3

            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)

    def test_tool_selection_add_mode_in_full_config(self):
        """mode: add with tools_db_path + top_k parses end-to-end."""
        temp_path = _write_config(
            [
                {
                    "type": "tool_selection",
                    "configuration": {
                        "enabled": True,
                        "mode": "add",
                        "tools_db_path": "config/tools_db.json",
                        "top_k": 5,
                        "similarity_threshold": 0.3,
                    },
                }
            ]
        )

        try:
            config = parse_user_config(temp_path)
            plugin = config.recipes[0].document.decisions[0].plugins[0]
            assert plugin.type.value == "tool_selection"
            assert plugin.configuration["mode"] == "add"
            assert plugin.configuration["tools_db_path"] == "config/tools_db.json"
            assert plugin.configuration["top_k"] == 5
            assert plugin.configuration["similarity_threshold"] == 0.3

            errors = validate_user_config(config)
            assert len(errors) == 0, f"Unexpected validation errors: {errors}"
        finally:
            os.unlink(temp_path)

    def test_tool_selection_invalid_mode_rejected(self):
        """mode outside {add, filter} is rejected by the Literal constraint."""
        with pytest.raises(PydanticValidationError):
            ToolSelectionPluginConfig(enabled=True, mode="bogus")

    def test_tool_selection_invalid_mode_in_full_config(self):
        """Invalid mode surfaces as a validator error referencing tool_selection."""
        temp_path = _write_config(
            [
                {
                    "type": "tool_selection",
                    "configuration": {"enabled": True, "mode": "bogus"},
                }
            ]
        )

        try:
            config = parse_user_config(temp_path)
            errors = validate_user_config(config)
            assert len(errors) > 0
            error_messages = [str(e) for e in errors]
            assert any("tool_selection" in msg.lower() for msg in error_messages)
        finally:
            os.unlink(temp_path)


class TestMultiplePlugins:
    """Test configurations with multiple plugins."""

    def test_multiple_plugins_in_decision(self):
        """Test decision with multiple plugins."""
        temp_path = _write_config(
            [
                {
                    "type": "router_replay",
                    "configuration": {"enabled": True, "max_records": 100},
                },
                {
                    "type": "response_cache",
                    "configuration": {
                        "enabled": True,
                        "similarity_threshold": 0.9,
                    },
                },
                {
                    "type": "system_prompt",
                    "configuration": {
                        "enabled": True,
                        "system_prompt": "You are a test assistant",
                    },
                },
            ]
        )

        try:
            config = parse_user_config(temp_path)
            plugins = config.recipes[0].document.decisions[0].plugins
            assert len(plugins) == 3

            plugin_types = [p.type.value for p in plugins]
            assert "router_replay" in plugin_types
            assert "response_cache" in plugin_types
            assert "system_prompt" in plugin_types

            errors = validate_user_config(config)
            assert len(errors) == 0
        finally:
            os.unlink(temp_path)

    def test_multiple_plugins_with_rag(self):
        """Test RAG alongside system_prompt, response_cache, and router_replay."""
        temp_path = _write_config(
            [
                {
                    "type": "system_prompt",
                    "configuration": {
                        "enabled": True,
                        "system_prompt": "You are a knowledge assistant.",
                    },
                },
                {
                    "type": "rag",
                    "configuration": {
                        "enabled": True,
                        "backend": "external_api",
                        "top_k": 5,
                        "similarity_threshold": 0.75,
                        "injection_mode": "tool_role",
                        "on_failure": "skip",
                        "cache_results": True,
                        "cache_ttl_seconds": 300,
                        "backend_config": {
                            "endpoint": "http://rag-service:8000/v1/search",
                            "request_format": "openai",
                            "timeout_seconds": 10,
                        },
                    },
                },
                {
                    "type": "response_cache",
                    "configuration": {
                        "enabled": True,
                        "similarity_threshold": 0.92,
                    },
                },
                {
                    "type": "router_replay",
                    "configuration": {"enabled": True, "max_records": 200},
                },
            ]
        )

        try:
            config = parse_user_config(temp_path)
            plugins = config.recipes[0].document.decisions[0].plugins
            assert len(plugins) == 4

            plugin_types = [p.type.value for p in plugins]
            assert "system_prompt" in plugin_types
            assert "rag" in plugin_types
            assert "response_cache" in plugin_types
            assert "router_replay" in plugin_types

            # Verify RAG plugin configuration is correctly parsed
            rag_plugin = next(p for p in plugins if p.type.value == "rag")
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
