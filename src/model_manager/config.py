"""
Configuration schema for Model Manager.

Defines dataclasses for model specifications and overall configuration.
"""

from dataclasses import dataclass, field
import os


@dataclass
class ModelSpec:
    """
    Specification for a single model to manage.

    Attributes:
        id: Unique identifier for this model (used for local directory name)
        repo_id: HuggingFace repository ID (e.g., "LLM-Semantic-Router/...")
        revision: Git revision (commit hash, tag, or branch). Defaults to "main"
        local_dir: Override local directory name. Defaults to using `id`
        files: Optional list of specific files to download. Downloads all if None
    """

    id: str
    repo_id: str
    revision: str = "main"
    local_dir: str | None = None
    files: list[str] | None = None

    def get_local_dir(self) -> str:
        """Get the local directory name for this model."""
        return self.local_dir if self.local_dir else self.id


@dataclass
class ModelsConfig:
    """
    Configuration for the Model Manager.

    Attributes:
        models: List of model specifications
        cache_dir: Directory to store downloaded models. Defaults to "models"
        verify: Verification level. Options: "none", "size", "sha256"
    """

    models: list[ModelSpec] = field(default_factory=list)
    cache_dir: str = "models"
    verify: str = "size"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_verify_levels = {"none", "size", "sha256"}
        if self.verify not in valid_verify_levels:
            raise ValueError(
                f"Invalid verify level '{self.verify}'. "
                f"Must be one of: {valid_verify_levels}"
            )
