"""Tests for model_manager.verifier module."""

import os
import pytest

from model_manager.verifier import (
    verify_model,
    verify_size,
    check_file_integrity,
    compute_sha256,
)


class TestVerifyModel:
    """Tests for verify_model function."""

    def test_verify_none_always_passes(self, temp_dir):
        """Test that verify level 'none' always passes."""
        assert verify_model("/nonexistent/path", "none") is True

    def test_verify_size_valid(self, mock_cached_model):
        """Test size verification on valid model."""
        assert verify_model(mock_cached_model, "size") is True

    def test_verify_sha256_valid(self, mock_cached_model):
        """Test SHA256 verification on valid model."""
        assert verify_model(mock_cached_model, "sha256") is True

    def test_verify_unknown_level_defaults_to_size(self, mock_cached_model):
        """Test that unknown verify level defaults to size check."""
        assert verify_model(mock_cached_model, "unknown") is True


class TestVerifySize:
    """Tests for verify_size function."""

    def test_verify_size_missing_directory(self):
        """Test that missing directory fails verification."""
        assert verify_size("/nonexistent/path") is False

    def test_verify_size_only_marker_files(self, temp_dir):
        """Test that directory with only marker files fails verification."""
        model_path = os.path.join(temp_dir, "empty-model")
        os.makedirs(model_path)

        # Create only marker files - no actual model content
        with open(os.path.join(model_path, ".downloaded"), "w") as f:
            f.write("2025-01-01T00:00:00Z\n")
        with open(os.path.join(model_path, ".gitattributes"), "w") as f:
            f.write("*.bin filter=lfs\n")

        # Should fail - no config.json or weight files
        assert verify_size(model_path) is False

    def test_verify_size_empty_file(self, temp_dir):
        """Test that empty file fails verification."""
        model_path = os.path.join(temp_dir, "empty-model")
        os.makedirs(model_path)

        # Create an empty non-marker file
        with open(os.path.join(model_path, "model.bin"), "w") as f:
            pass  # Empty file

        assert verify_size(model_path) is False

    def test_verify_size_valid_model(self, mock_cached_model):
        """Test that valid model passes size verification."""
        assert verify_size(mock_cached_model) is True

    def test_verify_size_allows_empty_marker(self, temp_dir):
        """Test that empty marker files are allowed."""
        model_path = os.path.join(temp_dir, "model-with-empty-marker")
        os.makedirs(model_path)

        # Empty marker file should be allowed
        with open(os.path.join(model_path, ".downloaded"), "w") as f:
            pass

        # But we also need real content
        with open(os.path.join(model_path, "config.json"), "w") as f:
            f.write('{"test": true}')

        assert verify_size(model_path) is True

    def test_verify_size_skips_lock_files(self, temp_dir):
        """Test that empty .lock files are skipped during verification."""
        model_path = os.path.join(temp_dir, "model-with-lock-files")
        os.makedirs(model_path)

        # Create valid content
        with open(os.path.join(model_path, "config.json"), "w") as f:
            f.write('{"test": true}')

        # Create empty .lock file (huggingface_hub creates these)
        with open(os.path.join(model_path, "tokenizer.json.lock"), "w") as f:
            pass  # Empty lock file should be skipped

        assert verify_size(model_path) is True

    def test_verify_size_skips_cache_directory(self, temp_dir):
        """Test that .cache directory files are skipped during verification."""
        model_path = os.path.join(temp_dir, "model-with-cache")
        cache_path = os.path.join(model_path, ".cache", "huggingface", "download")
        os.makedirs(cache_path)

        # Create valid content
        with open(os.path.join(model_path, "config.json"), "w") as f:
            f.write('{"test": true}')

        # Create empty file in .cache directory
        with open(os.path.join(cache_path, "empty.lock"), "w") as f:
            pass  # Should be skipped

        assert verify_size(model_path) is True


class TestCheckFileIntegrity:
    """Tests for check_file_integrity function."""

    def test_integrity_missing_directory(self):
        """Test that missing directory fails verification."""
        assert check_file_integrity("/nonexistent/path") is False

    def test_integrity_valid_model(self, mock_cached_model):
        """Test that valid model passes deep read check."""
        assert check_file_integrity(mock_cached_model) is True


class TestComputeSha256:
    """Tests for compute_sha256 function."""

    def test_compute_sha256_known_content(self, temp_dir):
        """Test SHA256 computation with known content."""
        from pathlib import Path

        # Known SHA256: echo -n "hello" | sha256sum
        # 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("hello")

        result = compute_sha256(test_file)
        assert (
            result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )

    def test_compute_sha256_binary_file(self, temp_dir):
        """Test SHA256 computation with binary content."""
        from pathlib import Path

        test_file = Path(temp_dir) / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        result = compute_sha256(test_file)
        assert len(result) == 64  # SHA256 hex is 64 chars
