"""Tests for model_manager.downloader module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from model_manager.downloader import download_model
from model_manager.errors import MissingModelError, DownloadError
from model_manager.config import ModelSpec


class TestDownloadModel:
    """Tests for download_model function."""

    def test_download_success(
        self, temp_dir, sample_model_spec, mock_snapshot_download
    ):
        """Test successful model download."""
        mock_snapshot_download.return_value = os.path.join(temp_dir, "test-model")

        # Create the mock directory with marker file
        model_path = os.path.join(temp_dir, "test-model")
        os.makedirs(model_path, exist_ok=True)

        result = download_model(sample_model_spec, temp_dir)

        mock_snapshot_download.assert_called_once()
        call_kwargs = mock_snapshot_download.call_args[1]
        assert call_kwargs["repo_id"] == sample_model_spec.repo_id
        assert call_kwargs["revision"] == sample_model_spec.revision
        assert call_kwargs["local_dir_use_symlinks"] is False

    def test_download_with_custom_local_dir(self, temp_dir, mock_snapshot_download):
        """Test download with custom local_dir."""
        spec = ModelSpec(
            id="test",
            repo_id="org/model",
            local_dir="custom-name",
        )

        mock_dir = os.path.join(temp_dir, "custom-name")
        os.makedirs(mock_dir, exist_ok=True)
        mock_snapshot_download.return_value = mock_dir

        download_model(spec, temp_dir)

        call_kwargs = mock_snapshot_download.call_args[1]
        assert call_kwargs["local_dir"].endswith("custom-name")

    def test_download_with_specific_files(self, temp_dir, mock_snapshot_download):
        """Test download with specific files."""
        spec = ModelSpec(
            id="test",
            repo_id="org/model",
            files=["config.json", "model.safetensors"],
        )

        mock_dir = os.path.join(temp_dir, "test")
        os.makedirs(mock_dir, exist_ok=True)
        mock_snapshot_download.return_value = mock_dir

        download_model(spec, temp_dir)

        call_kwargs = mock_snapshot_download.call_args[1]
        assert call_kwargs["allow_patterns"] == ["config.json", "model.safetensors"]
        assert call_kwargs["ignore_patterns"] is None

    def test_download_repo_not_found(self, temp_dir, sample_model_spec):
        """Test that missing repository raises MissingModelError."""
        from huggingface_hub.utils import RepositoryNotFoundError

        with patch("model_manager.downloader.snapshot_download") as mock:
            # Create a mock exception that inherits from RepositoryNotFoundError
            mock_response = MagicMock()
            mock_response.status_code = 404
            error = MagicMock(spec=RepositoryNotFoundError)
            error.__class__ = RepositoryNotFoundError
            mock.side_effect = RepositoryNotFoundError(
                "not found", response=mock_response
            )

            with pytest.raises(MissingModelError, match="Repository not found"):
                download_model(sample_model_spec, temp_dir)

    def test_download_revision_not_found(self, temp_dir, sample_model_spec):
        """Test that missing revision raises MissingModelError."""
        from huggingface_hub.utils import RevisionNotFoundError

        with patch("model_manager.downloader.snapshot_download") as mock:
            # Create a mock exception that inherits from RevisionNotFoundError
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock.side_effect = RevisionNotFoundError(
                "not found", response=mock_response
            )

            with pytest.raises(MissingModelError, match="Revision not found"):
                download_model(sample_model_spec, temp_dir)

    def test_download_generic_error(self, temp_dir, sample_model_spec):
        """Test that generic errors raise DownloadError."""
        with patch("model_manager.downloader.snapshot_download") as mock:
            mock.side_effect = Exception("network error")

            with pytest.raises(DownloadError, match="Failed to download"):
                download_model(sample_model_spec, temp_dir)
