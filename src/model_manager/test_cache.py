"""Tests for model_manager.cache module."""

import os
import pytest

from model_manager.cache import is_cached, get_model_path, clear_cache, get_cache_info
from model_manager.config import ModelSpec


class TestIsCached:
    """Tests for is_cached function."""

    def test_not_cached_no_directory(self, temp_dir, sample_model_spec):
        """Test that missing directory is not cached."""
        assert not is_cached(sample_model_spec, temp_dir)

    def test_not_cached_no_marker(self, temp_dir, sample_model_spec):
        """Test that directory without marker is not cached."""
        model_path = os.path.join(temp_dir, sample_model_spec.id)
        os.makedirs(model_path)

        # Create config.json but no marker
        with open(os.path.join(model_path, "config.json"), "w") as f:
            f.write("{}")

        assert not is_cached(sample_model_spec, temp_dir)

    def test_not_cached_no_model_files(self, temp_dir, sample_model_spec):
        """Test that directory with only marker is not cached."""
        model_path = os.path.join(temp_dir, sample_model_spec.id)
        os.makedirs(model_path)

        # Create marker but no model files
        with open(os.path.join(model_path, ".downloaded"), "w") as f:
            f.write("2025-01-01T00:00:00Z\n")

        assert not is_cached(sample_model_spec, temp_dir)

    def test_is_cached_valid(self, temp_dir, mock_cached_model, sample_model_spec):
        """Test that valid cached model is detected."""
        assert is_cached(sample_model_spec, temp_dir)


class TestGetModelPath:
    """Tests for get_model_path function."""

    def test_get_path_default(self, sample_model_spec):
        """Test getting path with default local_dir."""
        path = get_model_path(sample_model_spec, "models")
        assert path.endswith("models/test-model")

    def test_get_path_custom_local_dir(self):
        """Test getting path with custom local_dir."""
        spec = ModelSpec(
            id="test",
            repo_id="org/model",
            local_dir="custom-name",
        )
        path = get_model_path(spec, "cache")
        assert path.endswith("cache/custom-name")

    def test_get_path_is_absolute(self, sample_model_spec):
        """Test that returned path is absolute."""
        path = get_model_path(sample_model_spec, "models")
        assert os.path.isabs(path)


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clear_existing_cache(self, temp_dir, mock_cached_model, sample_model_spec):
        """Test clearing an existing cache."""
        assert os.path.exists(mock_cached_model)

        result = clear_cache(sample_model_spec, temp_dir)

        assert result is True
        assert not os.path.exists(mock_cached_model)

    def test_clear_nonexistent_cache(self, temp_dir, sample_model_spec):
        """Test clearing a non-existent cache returns False."""
        result = clear_cache(sample_model_spec, temp_dir)
        assert result is False


class TestGetCacheInfo:
    """Tests for get_cache_info function."""

    def test_get_info_cached(self, temp_dir, mock_cached_model, sample_model_spec):
        """Test getting cache info for cached model."""
        info = get_cache_info(sample_model_spec, temp_dir)

        assert info is not None
        assert info["path"] == mock_cached_model
        assert info["downloaded_at"] == "2025-01-01T00:00:00Z"
        assert info["size_bytes"] > 0
        assert info["file_count"] >= 3  # marker, config.json, model.safetensors

    def test_get_info_not_cached(self, temp_dir, sample_model_spec):
        """Test getting cache info for non-cached model."""
        info = get_cache_info(sample_model_spec, temp_dir)
        assert info is None
