"""Immutable, revision-keyed configuration for Decision inference backends.

Runtime profiles deliberately do not repeat a model ID or revision. The model
catalog owns both identities; its exact revision selects one packaged profile.
Keeping this module free of inference-framework imports makes catalog and
artifact validation safe in ordinary CLI processes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

RuntimeFamily = Literal["vela", "qwen3.5"]
RuntimeBackendName = Literal["rocm", "cuda", "cpu", "mlx"]
ChoiceNullDescriptionPolicy = Literal["render_key", "preserve_json_null"]

PROFILE_SCHEMA_VERSION = 2
# Initial benchmarked profile value, not a hard upper bound. A profile may tune
# it after backend/device correctness and performance validation.
DEFAULT_PHYSICAL_BATCH_SIZE = 8
_BACKEND_NAMES = ("rocm", "cuda", "cpu", "mlx")
_DTYPES = frozenset({"bfloat16", "float32"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class RuntimeProfileError(ValueError):
    """A packaged runtime profile is missing, invalid, or inconsistent."""


class UnsupportedRuntimeBackendError(RuntimeProfileError):
    """A profile has no qualified implementation for the requested backend."""


@dataclass(frozen=True, slots=True)
class ArtifactManifestIdentity:
    """Immutable identity of the model repository's artifact inventory."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactSelection:
    """Manifest-relative data files consumed by vLLM-SR-owned runtime code."""

    manifest: ArtifactManifestIdentity
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BackendQualification:
    """Evidence gate for one runtime implementation and its device targets."""

    qualified: bool
    targets: tuple[str, ...]
    backbone_dtype: str | None


@dataclass(frozen=True, slots=True)
class PromptPolicy:
    """Model-specific prompt rendering that must not be inferred by a backend."""

    choice_null_description: ChoiceNullDescriptionPolicy


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Validated backend configuration selected by an exact catalog revision."""

    revision: str
    family: RuntimeFamily
    artifact: ArtifactSelection
    max_input_tokens: int
    backbone_dtype: str
    head_dtype: str
    physical_batch_size: int
    temperature: float | None
    prompt_policy: PromptPolicy
    backends: Mapping[RuntimeBackendName, BackendQualification]

    def require_backend(
        self, backend: str, *, target: str | None = None
    ) -> BackendQualification:
        """Return a qualified backend or fail closed before model loading."""

        backend_name = backend.strip()
        qualification = self.backends.get(backend_name)  # type: ignore[arg-type]
        if qualification is None or not qualification.qualified:
            raise UnsupportedRuntimeBackendError(
                f"revision {self.revision} is not qualified for {backend_name!r}"
            )
        if target is not None and target not in qualification.targets:
            raise UnsupportedRuntimeBackendError(
                f"revision {self.revision} is not qualified for "
                f"{backend_name!r} target {target!r}"
            )
        return qualification


def validate_relative_artifact_path(value: object, *, field: str) -> str:
    """Return one canonical POSIX repository path with no traversal semantics."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeProfileError(f"{field} must be a nonblank canonical path")
    if "\\" in value or "\x00" in value or any(ord(char) < 32 for char in value):
        raise RuntimeProfileError(f"{field} contains an unsafe path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or value in {".", ".."}
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise RuntimeProfileError(f"{field} contains an unsafe path")
    return value


def load_runtime_profile(revision: str) -> RuntimeProfile:
    """Load a packaged profile whose filename is the exact catalog revision."""

    validate_catalog_revision(revision)
    resource = resources.files("decision_runtime.profiles").joinpath(f"{revision}.json")
    if not resource.is_file():
        raise RuntimeProfileError(
            f"no packaged Decision runtime profile for revision {revision}"
        )
    try:
        payload = resource.read_bytes()
    except OSError as error:
        raise RuntimeProfileError(
            f"could not read Decision runtime profile for revision {revision}"
        ) from error
    return parse_runtime_profile(payload, revision=revision)


def parse_runtime_profile(payload: bytes, *, revision: str) -> RuntimeProfile:
    """Parse profile bytes for package tests and alternate resource loaders."""

    validate_catalog_revision(revision)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeProfileError(
            "Decision runtime profile is not valid JSON"
        ) from error
    root = _mapping(document, "profile")
    _exact_keys(
        root,
        {
            "schema_version",
            "family",
            "artifact",
            "max_input_tokens",
            "dtype",
            "physical_batch_size",
            "calibration",
            "prompt_policy",
            "backends",
        },
        "profile",
    )
    if root["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise RuntimeProfileError("unsupported Decision runtime profile schema")

    family = root["family"]
    if family not in {"vela", "qwen3.5"}:
        raise RuntimeProfileError("profile.family is unsupported")

    artifact = _parse_artifact(root["artifact"])
    max_input_tokens = _positive_int(root["max_input_tokens"], "max_input_tokens")
    physical_batch_size = _positive_int(
        root["physical_batch_size"], "physical_batch_size"
    )

    dtype = _mapping(root["dtype"], "dtype")
    _exact_keys(dtype, {"backbone", "head"}, "dtype")
    backbone_dtype = _dtype(dtype["backbone"], "dtype.backbone")
    head_dtype = _dtype(dtype["head"], "dtype.head")
    temperature = _parse_calibration(root["calibration"])
    prompt_policy = _parse_prompt_policy(root["prompt_policy"])
    backends = _parse_backends(root["backends"])

    return RuntimeProfile(
        revision=revision,
        family=family,
        artifact=artifact,
        max_input_tokens=max_input_tokens,
        backbone_dtype=backbone_dtype,
        head_dtype=head_dtype,
        physical_batch_size=physical_batch_size,
        temperature=temperature,
        prompt_policy=prompt_policy,
        backends=backends,
    )


def _parse_artifact(value: object) -> ArtifactSelection:
    artifact = _mapping(value, "artifact")
    _exact_keys(artifact, {"manifest", "files"}, "artifact")
    manifest = _mapping(artifact["manifest"], "artifact.manifest")
    _exact_keys(manifest, {"path", "sha256", "size_bytes"}, "artifact.manifest")
    manifest_path = validate_relative_artifact_path(
        manifest["path"], field="artifact.manifest.path"
    )
    manifest_sha256 = _digest(manifest["sha256"], "artifact.manifest.sha256")
    manifest_size = _positive_int(
        manifest["size_bytes"], "artifact.manifest.size_bytes"
    )

    raw_files = artifact["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise RuntimeProfileError("artifact.files must be a non-empty list")
    selected = tuple(
        validate_relative_artifact_path(item, field="artifact.files[]")
        for item in raw_files
    )
    if len(set(selected)) != len(selected):
        raise RuntimeProfileError("artifact.files contains duplicate paths")
    return ArtifactSelection(
        manifest=ArtifactManifestIdentity(
            path=manifest_path,
            sha256=manifest_sha256,
            size_bytes=manifest_size,
        ),
        files=selected,
    )


def _parse_calibration(value: object) -> float | None:
    if value is None:
        return None
    calibration = _mapping(value, "calibration")
    _exact_keys(calibration, {"temperature"}, "calibration")
    temperature = calibration["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise RuntimeProfileError("calibration.temperature must be positive and finite")
    return float(temperature)


def _parse_prompt_policy(value: object) -> PromptPolicy:
    policy = _mapping(value, "prompt_policy")
    _exact_keys(policy, {"choice_null_description"}, "prompt_policy")
    choice_null_description = policy["choice_null_description"]
    if choice_null_description not in {"render_key", "preserve_json_null"}:
        raise RuntimeProfileError(
            "prompt_policy.choice_null_description is unsupported"
        )
    return PromptPolicy(choice_null_description=choice_null_description)


def _parse_backends(
    value: object,
) -> Mapping[RuntimeBackendName, BackendQualification]:
    backends = _mapping(value, "backends")
    _exact_keys(backends, set(_BACKEND_NAMES), "backends")
    parsed: dict[RuntimeBackendName, BackendQualification] = {}
    for name in _BACKEND_NAMES:
        raw = _mapping(backends[name], f"backends.{name}")
        _exact_keys(
            raw, {"qualified", "targets", "backbone_dtype"}, f"backends.{name}"
        )
        qualified = raw["qualified"]
        targets = raw["targets"]
        dtype = raw["backbone_dtype"]
        if not isinstance(qualified, bool):
            raise RuntimeProfileError(f"backends.{name}.qualified must be a boolean")
        if (
            not isinstance(targets, list)
            or any(not isinstance(item, str) or not item.strip() for item in targets)
            or len(targets) != len(set(targets))
        ):
            raise RuntimeProfileError(
                f"backends.{name}.targets must contain unique nonblank strings"
            )
        if qualified != bool(targets):
            raise RuntimeProfileError(
                f"backends.{name} must list targets exactly when it is qualified"
            )
        if dtype is not None:
            dtype = _dtype(dtype, f"backends.{name}.backbone_dtype")
        if qualified and dtype is None:
            raise RuntimeProfileError(
                f"backends.{name}.backbone_dtype is required when qualified"
            )
        parsed[name] = BackendQualification(  # type: ignore[literal-required]
            qualified=qualified,
            targets=tuple(targets),
            backbone_dtype=dtype,
        )
    return MappingProxyType(parsed)


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeProfileError(f"{field} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise RuntimeProfileError(f"{field} fields do not match the runtime contract")


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeProfileError(f"{field} must be a positive integer")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeProfileError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _dtype(value: object, field: str) -> str:
    if value not in _DTYPES:
        raise RuntimeProfileError(f"{field} is unsupported")
    return str(value)


def validate_catalog_revision(revision: str) -> str:
    """Require an immutable, full lower-case Git revision and return it."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise RuntimeProfileError("catalog revision must be a full lowercase Git SHA")
    return revision
