"""
Downloader module for HuggingFace model downloads.

Uses huggingface_hub's snapshot_download for reliable downloads with
resume support, progress tracking, and proper caching.
"""

import logging
import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from .config import ModelSpec
from .errors import DownloadError, MissingModelError

logger = logging.getLogger(__name__)


def download_model(spec: ModelSpec, cache_dir: str) -> str:
    """
    Download a model from HuggingFace using snapshot_download.

    Args:
        spec: Model specification
        cache_dir: Base directory to store models

    Returns:
        Local path to the downloaded model

    Raises:
        MissingModelError: If the model or revision doesn't exist
        DownloadError: If download fails for other reasons
    """
    local_dir = os.path.join(cache_dir, spec.get_local_dir())

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(
        f"Downloading model '{spec.id}' from '{spec.repo_id}' "
        f"(revision: {spec.revision}) to '{local_dir}'"
    )

    try:
        result_path = snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            # Only download specific files if specified
            allow_patterns=spec.files if spec.files else None,
            # Common patterns to ignore for ML models
            ignore_patterns=(
                [
                    "*.onnx",
                    "*.onnx_data",
                    "*.msgpack",
                    "*.h5",
                    "*.tflite",
                    "*.ot",
                ]
                if not spec.files
                else None
            ),
        )

        # Write a marker file with download timestamp
        marker_path = os.path.join(result_path, ".downloaded")
        with open(marker_path, "w") as f:
            from datetime import datetime, timezone

            f.write(datetime.now(timezone.utc).isoformat() + "\n")

        logger.info(f"Successfully downloaded model '{spec.id}' to '{result_path}'")
        return result_path

    except RepositoryNotFoundError:
        raise MissingModelError(
            f"Repository not found: '{spec.repo_id}'. "
            "Check if the repository exists and you have access."
        )
    except RevisionNotFoundError:
        raise MissingModelError(
            f"Revision not found: '{spec.revision}' in repository '{spec.repo_id}'. "
            "Check if the revision (commit/tag/branch) exists."
        )
    except Exception as e:
        raise DownloadError(
            f"Failed to download model '{spec.id}' from '{spec.repo_id}': {e}"
        ) from e
