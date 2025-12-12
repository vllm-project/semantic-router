"""Tests for model_manager.registry module."""

import os
import pytest

from model_manager.registry import load_models_config, parse_config, parse_model_spec
from model_manager.errors import ConfigurationError


class TestLoadModelsConfig:
    """Tests for load_models_config function."""

    def test_load_valid_config(self, sample_config_yaml):
        """Test loading a valid configuration file."""
        config = load_models_config(sample_config_yaml)

        assert config.cache_dir == "models"
        assert config.verify == "size"
        assert len(config.models) == 2

        # Check first model
        assert config.models[0].id == "test-model"
        assert config.models[0].repo_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.models[0].revision == "main"

        # Check second model with custom local_dir
        assert config.models[1].id == "custom-model"
        assert config.models[1].local_dir == "my-custom-model"
        assert config.models[1].revision == "v1.0.0"

    def test_load_missing_config(self, temp_dir):
        """Test that missing config file raises error."""
        with pytest.raises(ConfigurationError, match="not found"):
            load_models_config(os.path.join(temp_dir, "nonexistent.yaml"))

    def test_load_invalid_yaml(self, temp_dir):
        """Test that invalid YAML raises error."""
        config_path = os.path.join(temp_dir, "invalid.yaml")
        with open(config_path, "w") as f:
            f.write("{ invalid yaml: [")

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_models_config(config_path)

    def test_load_empty_config(self, temp_dir):
        """Test that empty config file raises error."""
        config_path = os.path.join(temp_dir, "empty.yaml")
        with open(config_path, "w") as f:
            f.write("")

        with pytest.raises(ConfigurationError, match="Empty configuration"):
            load_models_config(config_path)


class TestParseConfig:
    """Tests for parse_config function."""

    def test_parse_minimal_config(self):
        """Test parsing minimal config with defaults."""
        data = {"models": [{"id": "test", "repo_id": "org/model"}]}
        config = parse_config(data)

        assert config.cache_dir == "models"  # default
        assert config.verify == "size"  # default
        assert len(config.models) == 1

    def test_parse_full_config(self):
        """Test parsing config with all options."""
        data = {
            "cache_dir": "/custom/cache",
            "verify": "sha256",
            "models": [
                {
                    "id": "test",
                    "repo_id": "org/model",
                    "revision": "v1.0",
                    "local_dir": "custom-name",
                    "files": ["config.json", "model.safetensors"],
                }
            ],
        }
        config = parse_config(data)

        assert config.cache_dir == "/custom/cache"
        assert config.verify == "sha256"
        assert config.models[0].files == ["config.json", "model.safetensors"]


class TestParseModelSpec:
    """Tests for parse_model_spec function."""

    def test_parse_minimal_spec(self):
        """Test parsing model spec with required fields only."""
        data = {"id": "test", "repo_id": "org/model"}
        spec = parse_model_spec(data)

        assert spec.id == "test"
        assert spec.repo_id == "org/model"
        assert spec.revision == "main"  # default
        assert spec.local_dir is None
        assert spec.files is None

    def test_parse_full_spec(self):
        """Test parsing model spec with all fields."""
        data = {
            "id": "test",
            "repo_id": "org/model",
            "revision": "abc123",
            "local_dir": "my-model",
            "files": ["file1.txt"],
        }
        spec = parse_model_spec(data)

        assert spec.id == "test"
        assert spec.revision == "abc123"
        assert spec.local_dir == "my-model"
        assert spec.files == ["file1.txt"]

    def test_parse_missing_id(self):
        """Test that missing id raises error."""
        with pytest.raises(ConfigurationError, match="missing required 'id'"):
            parse_model_spec({"repo_id": "org/model"})

    def test_parse_missing_repo_id(self):
        """Test that missing repo_id raises error."""
        with pytest.raises(ConfigurationError, match="missing required 'repo_id'"):
            parse_model_spec({"id": "test"})
