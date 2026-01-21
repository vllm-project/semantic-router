"""Tests for YAML generation with plugins."""

import tempfile
import os
import yaml

from cli.parser import parse_user_config
from cli.merger import merge_configs
from cli.defaults import load_embedded_defaults


class TestPluginYAMLGeneration:
    """Test YAML generation with plugins."""

    def test_router_replay_plugin_in_generated_yaml(self):
        """Test that router_replay plugin is correctly serialized in generated YAML."""
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
            defaults = load_embedded_defaults()
            merged = merge_configs(config, defaults)

            # Check that plugins are in merged config
            assert "decisions" in merged
            assert len(merged["decisions"]) > 0

            decision = merged["decisions"][0]
            assert "plugins" in decision
            assert len(decision["plugins"]) > 0

            # Find router_replay plugin
            router_replay = next(
                (p for p in decision["plugins"] if p["type"] == "router_replay"), None
            )
            assert router_replay is not None
            assert router_replay["configuration"]["enabled"] is True
            assert router_replay["configuration"]["max_records"] == 100
            assert router_replay["configuration"]["capture_request_body"] is True
            assert router_replay["configuration"]["capture_response_body"] is False
            assert router_replay["configuration"]["max_body_bytes"] == 2048

            # Test YAML serialization
            yaml_output = yaml.dump(merged, default_flow_style=False, sort_keys=False)
            assert "router_replay" in yaml_output
            assert "max_records: 100" in yaml_output
            assert "capture_request_body: true" in yaml_output
        finally:
            os.unlink(temp_path)

    def test_multiple_plugins_in_generated_yaml(self):
        """Test that multiple plugins are correctly serialized."""
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
          system_prompt: "Test prompt"
          mode: "replace"
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
            defaults = load_embedded_defaults()
            merged = merge_configs(config, defaults)

            decision = merged["decisions"][0]
            plugins = decision["plugins"]

            assert len(plugins) == 3

            plugin_types = [p["type"] for p in plugins]
            assert "router_replay" in plugin_types
            assert "semantic-cache" in plugin_types
            assert "system_prompt" in plugin_types

            # Verify each plugin's configuration
            router_replay = next(
                (p for p in plugins if p["type"] == "router_replay"), None
            )
            assert router_replay["configuration"]["enabled"] is True

            semantic_cache = next(
                (p for p in plugins if p["type"] == "semantic-cache"), None
            )
            assert semantic_cache["configuration"]["enabled"] is True
            assert semantic_cache["configuration"]["similarity_threshold"] == 0.9

            system_prompt = next(
                (p for p in plugins if p["type"] == "system_prompt"), None
            )
            assert system_prompt["configuration"]["enabled"] is True
            assert system_prompt["configuration"]["system_prompt"] == "Test prompt"
            assert system_prompt["configuration"]["mode"] == "replace"

            # Test YAML serialization
            yaml_output = yaml.dump(merged, default_flow_style=False, sort_keys=False)
            assert "router_replay" in yaml_output
            assert "semantic-cache" in yaml_output
            assert "system_prompt" in yaml_output
        finally:
            os.unlink(temp_path)

    def test_plugin_configuration_preserved(self):
        """Test that all plugin configuration fields are preserved in generated YAML."""
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
          max_records: 200
          capture_request_body: true
          capture_response_body: true
          max_body_bytes: 8192
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
            defaults = load_embedded_defaults()
            merged = merge_configs(config, defaults)

            decision = merged["decisions"][0]
            router_replay = next(
                (p for p in decision["plugins"] if p["type"] == "router_replay"), None
            )

            # Verify all fields are preserved
            config_dict = router_replay["configuration"]
            assert config_dict["enabled"] is True
            assert config_dict["max_records"] == 200
            assert config_dict["capture_request_body"] is True
            assert config_dict["capture_response_body"] is True
            assert config_dict["max_body_bytes"] == 8192

            # Verify YAML output contains all fields
            yaml_output = yaml.dump(merged, default_flow_style=False, sort_keys=False)
            assert "max_records: 200" in yaml_output
            assert "capture_request_body: true" in yaml_output
            assert "capture_response_body: true" in yaml_output
            assert "max_body_bytes: 8192" in yaml_output
        finally:
            os.unlink(temp_path)
