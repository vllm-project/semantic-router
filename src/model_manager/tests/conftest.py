"""Pytest fixtures for model_manager tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_manager.config import ModelSpec, ModelsConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model_spec():
    """Create a sample ModelSpec for testing."""
    return ModelSpec(
        id="test-model",
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )


@pytest.fixture
def sample_config(sample_model_spec):
    """Create a sample ModelsConfig for testing."""
    return ModelsConfig(
        models=[sample_model_spec],
        cache_dir="models",
        verify="size",
    )


@pytest.fixture
def sample_config_yaml(temp_dir):
    """Create a sample models.yaml file for testing."""
    config_path = os.path.join(temp_dir, "models.yaml")
    config_content = """
cache_dir: "models"
verify: "size"

models:
  - id: test-model
    repo_id: sentence-transformers/all-MiniLM-L6-v2
    revision: main

  - id: custom-model
    repo_id: custom/custom-model
    revision: v1.0.0
    local_dir: my-custom-model
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


@pytest.fixture
def mock_cached_model(temp_dir, sample_model_spec):
    """Create a mock cached model directory."""
    model_path = os.path.join(temp_dir, sample_model_spec.id)
    os.makedirs(model_path)

    # Create marker file
    with open(os.path.join(model_path, ".downloaded"), "w") as f:
        f.write("2025-01-01T00:00:00Z\n")

    # Create mock model files
    with open(os.path.join(model_path, "config.json"), "w") as f:
        f.write('{"model_type": "bert"}')

    with open(os.path.join(model_path, "model.safetensors"), "wb") as f:
        f.write(b"mock model weights" * 100)

    return model_path


@pytest.fixture
def mock_snapshot_download():
    """Mock huggingface_hub.snapshot_download."""
    with patch("model_manager.downloader.snapshot_download") as mock:
        mock.return_value = "/tmp/mock-model-path"
        yield mock
