"""
Custom exception types for Model Manager.
"""


class ModelManagerError(Exception):
    """Base exception for all Model Manager errors."""

    pass


class MissingModelError(ModelManagerError):
    """Raised when a requested model is not found in configuration."""

    pass


class BadChecksumError(ModelManagerError):
    """Raised when model verification fails (file missing, wrong size, or bad hash)."""

    pass


class DownloadError(ModelManagerError):
    """Raised when model download fails."""

    pass


class ConfigurationError(ModelManagerError):
    """Raised when configuration is invalid or missing."""

    pass
