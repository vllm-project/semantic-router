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
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

from .runtime_limits import MAX_JOB_TURN_BATCHES, MAX_PENDING_ROWS

RuntimeFamily = Literal["vela", "qwen3.5"]
ChoiceNullDescriptionPolicy = Literal["render_key", "preserve_json_null"]
QwenGatedDeltaKernel = Literal["native_torch", "accelerated"]

PROFILE_SCHEMA_VERSION = 4
# Initial benchmarked profile value, not a hard upper bound. A profile may tune
# it after backend/device correctness and performance validation.
DEFAULT_PHYSICAL_BATCH_SIZE = 8
SHORT_GRAPH_PHYSICAL_BATCH_SIZE = 8
_MAX_GRAPH_PREWARM_SHAPES = 2
_MAX_GRAPH_PADDED_TOKENS = 256
_GRAPH_PADDED_TOKEN_MULTIPLE = 32
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
class BackendExecutionPolicy:
    """Model-selected optimizations for one installed hardware backend."""

    backbone_graph: Literal["short_b8"] | None = None
    graph_prewarm_padded_tokens: tuple[int, ...] = ()
    gated_delta: QwenGatedDeltaKernel = "accelerated"
    max_physical_batch_size: int | None = None
    job_turn_batches: int | None = None


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
    execution: Mapping[str, BackendExecutionPolicy]

    def validate_physical_batch_size(self, backend: str, size: int) -> None:
        """Reject a launch shape outside shared or backend-specific limits."""

        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 1 <= size <= MAX_PENDING_ROWS
        ):
            raise RuntimeProfileError(
                f"physical batch size must be between 1 and {MAX_PENDING_ROWS}"
            )
        policy = self.execution.get(backend)
        maximum = policy.max_physical_batch_size if policy is not None else None
        if maximum is not None and size > maximum:
            raise RuntimeProfileError(
                f"{backend} physical batch size {size} exceeds profile maximum {maximum}"
            )

    def use_short_b8_graph(self, backend: str, physical_batch_size: int) -> bool:
        """Select a profiled acceleration only for its measured batch shape."""

        policy = self.execution.get(backend)
        return (
            policy is not None
            and policy.backbone_graph == "short_b8"
            and physical_batch_size == SHORT_GRAPH_PHYSICAL_BATCH_SIZE
        )

    def rows_per_job_turn(self, backend: str, physical_batch_size: int) -> int:
        """Keep the default fair row rotation unless a backend is profiled."""

        policy = self.execution.get(backend)
        if policy is None or policy.job_turn_batches is None:
            return 1
        return physical_batch_size * policy.job_turn_batches


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
    required = {
        "schema_version",
        "family",
        "artifact",
        "max_input_tokens",
        "dtype",
        "physical_batch_size",
        "calibration",
        "prompt_policy",
    }
    if set(root) not in (required, required | {"execution"}):
        raise RuntimeProfileError("profile fields do not match the runtime contract")
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
    execution = (
        _parse_execution(root["execution"], family=family)
        if "execution" in root
        else MappingProxyType({})
    )

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
        execution=execution,
    )


def _parse_execution(
    value: object, *, family: RuntimeFamily
) -> Mapping[str, BackendExecutionPolicy]:
    execution = _mapping(value, "execution")
    if set(execution) != {"rocm"} or family != "qwen3.5":
        raise RuntimeProfileError("execution backend or model family is unsupported")
    rocm = _mapping(execution["rocm"], "execution.rocm")
    allowed = {
        "backbone_graph",
        "graph_prewarm_padded_tokens",
        "gated_delta",
        "max_physical_batch_size",
        "job_turn_batches",
    }
    if not rocm or not set(rocm) <= allowed:
        raise RuntimeProfileError(
            "execution.rocm fields do not match the runtime contract"
        )
    graph = rocm.get("backbone_graph")
    if "backbone_graph" in rocm and graph != "short_b8":
        raise RuntimeProfileError("execution.rocm.backbone_graph is unsupported")
    prewarm = rocm.get("graph_prewarm_padded_tokens", [])
    if (
        not isinstance(prewarm, list)
        or len(prewarm) > _MAX_GRAPH_PREWARM_SHAPES
        or any(
            type(tokens) is not int
            or not _GRAPH_PADDED_TOKEN_MULTIPLE <= tokens <= _MAX_GRAPH_PADDED_TOKENS
            or tokens % _GRAPH_PADDED_TOKEN_MULTIPLE
            for tokens in prewarm
        )
        or len(set(prewarm)) != len(prewarm)
        or ("graph_prewarm_padded_tokens" in rocm and graph != "short_b8")
    ):
        raise RuntimeProfileError(
            "execution.rocm.graph_prewarm_padded_tokens is unsupported"
        )
    kernel = rocm.get("gated_delta", "accelerated")
    if not isinstance(kernel, str) or kernel not in {"native_torch", "accelerated"}:
        raise RuntimeProfileError("execution.rocm.gated_delta is unsupported")
    maximum = rocm.get("max_physical_batch_size")
    if kernel == "native_torch":
        maximum = _positive_int(maximum, "execution.rocm.max_physical_batch_size")
    elif maximum is not None:
        raise RuntimeProfileError(
            "execution.rocm.max_physical_batch_size requires native_torch"
        )
    if graph is not None and kernel == "native_torch":
        raise RuntimeProfileError(
            "execution.rocm cannot combine short_b8 graph with native_torch"
        )
    job_turn_batches = rocm.get("job_turn_batches")
    if job_turn_batches is not None and (
        type(job_turn_batches) is not int
        or not 1 <= job_turn_batches <= MAX_JOB_TURN_BATCHES
    ):
        raise RuntimeProfileError(
            f"execution.rocm.job_turn_batches must be 1 to {MAX_JOB_TURN_BATCHES}"
        )
    return MappingProxyType(
        {
            "rocm": BackendExecutionPolicy(
                backbone_graph=graph,
                graph_prewarm_padded_tokens=tuple(prewarm),
                gated_delta=kernel,
                max_physical_batch_size=maximum,
                job_turn_batches=job_turn_batches,
            )
        }
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
