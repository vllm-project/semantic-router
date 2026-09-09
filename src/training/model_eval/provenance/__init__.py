"""Versioned provenance manifests for built-in Router Models.

The package defines one machine-readable contract covering the four objects a
Router Model passes through: the dataset it was trained and measured on, the
training run that produced it, the artifact that run emitted, and the
evaluation that measured that artifact.

Public surface:
    load_manifest        -- parse and schema-validate a single manifest file
    validate_bundle      -- schema + cross-reference validation over a directory
    ManifestError        -- raised for every validation failure
"""

from .crossref import validate_bundle
from .manifest import (
    MANIFEST_KINDS,
    SCHEMA_VERSION,
    ManifestError,
    load_manifest,
    load_manifests,
)

__all__ = [
    "MANIFEST_KINDS",
    "SCHEMA_VERSION",
    "ManifestError",
    "load_manifest",
    "load_manifests",
    "validate_bundle",
]
