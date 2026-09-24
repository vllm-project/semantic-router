"""Model-keyed implementation profiles for Decision inference backends.

Runtime profiles deliberately do not repeat a model ID or revision. The model
catalog owns both identities; the model name selects one packaged template and
its exact revision selects the artifact snapshot.
Keeping this module free of inference-framework imports makes catalog and
artifact validation safe in ordinary CLI processes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from typing import Any, Literal

RuntimeFamily = Literal["vela", "qwen3.5"]
ChoiceNullDescriptionPolicy = Literal["render_key", "preserve_json_null"]

PROFILE_SCHEMA_VERSION = 4
# Initial benchmarked profile value, not a hard upper bound. A profile may tune
# it after backend/device correctness and performance validation.
DEFAULT_PHYSICAL_BATCH_SIZE = 8
_PROFILE_FAMILY_DIRECTORIES = ("vela", "qwen35")
_DTYPES = frozenset({"bfloat16", "float32"})
_ASCII_CONTROL_LIMIT = 32
_REVISION = re.compile(r"[0-9a-f]{40}")


class RuntimeProfileError(ValueError):
    """A packaged runtime profile is missing, invalid, or inconsistent."""


class UnsupportedRuntimeBackendError(RuntimeProfileError):
    """A profile has no qualified implementation for the requested backend."""


@dataclass(frozen=True, slots=True)
class ArtifactManifestIdentity:
    """Observed manifest identity from one selected immutable snapshot."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactSelection:
    """Manifest location for one model-family layout.

    ``files`` is an explicit selection seam for alternate in-memory test
    layouts. Packaged profiles select files from the snapshot manifest instead.
    """

    manifest_path: str
    files: tuple[str, ...] = ()


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


def validate_relative_artifact_path(value: object, *, field: str) -> str:
    """Return one canonical POSIX repository path with no traversal semantics."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeProfileError(f"{field} must be a nonblank canonical path")
    if (
        "\\" in value
        or "\x00" in value
        or any(ord(char) < _ASCII_CONTROL_LIMIT for char in value)
    ):
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


def load_runtime_profile(profile_id: str, *, revision: str) -> RuntimeProfile:
    """Load a stable model template and bind it to one catalog revision."""

    validate_catalog_revision(revision)
    if (
        not isinstance(profile_id, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", profile_id) is None
    ):
        raise RuntimeProfileError("Decision runtime profile ID is invalid")
    root = resources.files("decision_runtime.profiles")
    matches = tuple(
        (directory, resource)
        for directory in _PROFILE_FAMILY_DIRECTORIES
        if (resource := root.joinpath(directory, f"{profile_id}.json")).is_file()
    )
    if len(matches) != 1:
        raise RuntimeProfileError(
            f"expected one packaged Decision runtime profile for model {profile_id}"
        )
    try:
        payload = matches[0][1].read_bytes()
    except OSError as error:
        raise RuntimeProfileError(
            f"could not read Decision runtime profile for model {profile_id}"
        ) from error
    profile = parse_runtime_profile(payload, revision=revision)
    directory = "vela" if profile.family == "vela" else "qwen35"
    if matches[0][0] != directory:
        raise RuntimeProfileError(
            "Decision profile is in the wrong model-family directory"
        )
    return profile


def parse_runtime_profile(payload: bytes, *, revision: str) -> RuntimeProfile:
    """Parse profile bytes for package tests and alternate resource loaders."""

    validate_catalog_revision(revision)
    try:
        document = json.loads(payload)
    except (TypeError, ValueError, RecursionError) as error:
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
        },
        "profile",
    )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != PROFILE_SCHEMA_VERSION
    ):
        raise RuntimeProfileError("unsupported Decision runtime profile schema")

    family = root["family"]
    if not isinstance(family, str) or family not in {"vela", "qwen3.5"}:
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
    )


def _parse_artifact(value: object) -> ArtifactSelection:
    artifact = _mapping(value, "artifact")
    _exact_keys(artifact, {"manifest_path"}, "artifact")
    manifest_path = validate_relative_artifact_path(
        artifact["manifest_path"], field="artifact.manifest_path"
    )
    return ArtifactSelection(manifest_path=manifest_path)


def _parse_calibration(value: object) -> float | None:
    if value is None:
        return None
    calibration = _mapping(value, "calibration")
    _exact_keys(calibration, {"temperature"}, "calibration")
    temperature = calibration["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise RuntimeProfileError("calibration.temperature must be positive and finite")
    try:
        temperature_value = float(temperature)
    except OverflowError as error:
        raise RuntimeProfileError(
            "calibration.temperature must be positive and finite"
        ) from error
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise RuntimeProfileError("calibration.temperature must be positive and finite")
    return temperature_value


def _parse_prompt_policy(value: object) -> PromptPolicy:
    policy = _mapping(value, "prompt_policy")
    _exact_keys(policy, {"choice_null_description"}, "prompt_policy")
    choice_null_description = policy["choice_null_description"]
    if not isinstance(choice_null_description, str) or choice_null_description not in {
        "render_key",
        "preserve_json_null",
    }:
        raise RuntimeProfileError(
            "prompt_policy.choice_null_description is unsupported"
        )
    return PromptPolicy(choice_null_description=choice_null_description)


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


def _dtype(value: object, field: str) -> str:
    if not isinstance(value, str) or value not in _DTYPES:
        raise RuntimeProfileError(f"{field} is unsupported")
    return value


def validate_catalog_revision(revision: str) -> str:
    """Require an immutable, full lower-case Git revision and return it."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise RuntimeProfileError("catalog revision must be a full lowercase Git SHA")
    return revision
