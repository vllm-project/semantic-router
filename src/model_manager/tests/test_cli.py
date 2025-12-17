"""Tests for model_manager.cli module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from model_manager.cli import get_default_config, main


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_default_config_without_env(self):
        """Test that default config is returned when env not set."""
        # Use patch.dict without clear=True to avoid race conditions
        # Explicitly set CI_MINIMAL_MODELS to empty string if it exists, or ensure it's not set
        # But safest is to just mock os.environ.get
        with patch("os.environ.get") as mock_get:
            mock_get.return_value = ""
            result = get_default_config()
            assert result == "config/model_manager/models.yaml"

    def test_minimal_config_with_true(self):
        """Test that minimal config is returned when CI_MINIMAL_MODELS=true."""
        with patch.dict(os.environ, {"CI_MINIMAL_MODELS": "true"}):
            result = get_default_config()
            assert result == "config/model_manager/models.minimal.yaml"

    def test_minimal_config_with_1(self):
        """Test that minimal config is returned when CI_MINIMAL_MODELS=1."""
        with patch.dict(os.environ, {"CI_MINIMAL_MODELS": "1"}):
            result = get_default_config()
            assert result == "config/model_manager/models.minimal.yaml"

    def test_minimal_config_with_yes(self):
        """Test that minimal config is returned when CI_MINIMAL_MODELS=yes."""
        with patch.dict(os.environ, {"CI_MINIMAL_MODELS": "yes"}):
            result = get_default_config()
            assert result == "config/model_manager/models.minimal.yaml"

    def test_minimal_config_case_insensitive(self):
        """Test that CI_MINIMAL_MODELS check is case insensitive."""
        with patch.dict(os.environ, {"CI_MINIMAL_MODELS": "TRUE"}):
            result = get_default_config()
            assert result == "config/model_manager/models.minimal.yaml"

    def test_default_config_with_false(self):
        """Test that default config is returned when CI_MINIMAL_MODELS=false."""
        with patch.dict(os.environ, {"CI_MINIMAL_MODELS": "false"}):
            result = get_default_config()
            assert result == "config/model_manager/models.yaml"


class TestMainCLI:
    """Tests for main CLI function."""

    def test_main_missing_config(self):
        """Test that missing config file returns error code."""
        with patch(
            "sys.argv", ["model_manager", "--config", "/nonexistent/config.yaml"]
        ):
            result = main()
            assert result == 1

    def test_main_list_models(self, sample_config_yaml, capsys):
        """Test --list flag outputs model information."""
        with patch(
            "sys.argv", ["model_manager", "--config", sample_config_yaml, "--list"]
        ):
            result = main()
            assert result == 0

            captured = capsys.readouterr()
            assert "test-model" in captured.out
            assert "custom-model" in captured.out

    def test_main_verify_only_no_models(self, sample_config_yaml):
        """Test --verify-only with no cached models."""
        with patch(
            "sys.argv",
            ["model_manager", "--config", sample_config_yaml, "--verify-only"],
        ):
            result = main()
            # Returns 1 because models aren't cached
            assert result == 1

    def test_main_clean_no_cached_models(self, sample_config_yaml, capsys):
        """Test --clean with no cached models."""
        with patch(
            "sys.argv", ["model_manager", "--config", sample_config_yaml, "--clean"]
        ):
            result = main()
            assert result == 0

            # Note: log messages go to stderr, not stdout
            captured = capsys.readouterr()
            # Just verify it returns 0 - log message format may vary

    def test_main_clean_model_not_found(self, sample_config_yaml):
        """Test --clean-model with non-existent model ID."""
        with patch(
            "sys.argv",
            [
                "model_manager",
                "--config",
                sample_config_yaml,
                "--clean-model",
                "nonexistent",
            ],
        ):
            result = main()
            assert result == 1

    def test_main_keyboard_interrupt(self, sample_config_yaml):
        """Test that KeyboardInterrupt is handled gracefully."""
        with patch("sys.argv", ["model_manager", "--config", sample_config_yaml]):
            with patch("model_manager.cli.ModelManager.from_config") as mock:
                mock.side_effect = KeyboardInterrupt()
                result = main()
                assert result == 130
