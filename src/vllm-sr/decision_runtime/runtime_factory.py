"""Assemble one pinned Decision artifact into a resident HTTP backend."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .artifacts import ArtifactError, VerifiedArtifact, open_verified_artifact
from .backend import ModelDescriptor
from .catalog_adapter import (
    ResolvedRuntimeModel,
    RuntimeModelResolutionError,
    resolve_decision_runtime_model,
)
from .physical_batching import PhysicalBatchBackend
from .row_executor import TorchDecisionRowExecutor
from .runtime_profile import RuntimeProfileError
from .scheduler import ModelScheduler

MAX_PENDING_ROWS = 4096


class RuntimeAssemblyError(RuntimeError):
    """Launch inputs cannot safely identify and load one resident model."""


@dataclass(frozen=True, slots=True)
class RuntimeLaunchConfig:
    model: str
    revision: str
    backend: Literal["cpu", "rocm", "cuda"]
    artifact_root: Path
    artifact_content_id: str
    host: str
    port: int
    max_batch: int
    max_concurrency: int
    max_queue: int


@dataclass(frozen=True, slots=True)
class AssembledRuntime:
    backend: PhysicalBatchBackend
    scheduler: ModelScheduler


def assemble_runtime(config: RuntimeLaunchConfig) -> AssembledRuntime:
    """Verify identity and bytes before loading any model framework."""

    _validate_config(config)
    try:
        model = resolve_decision_runtime_model(config.model, backend=config.backend)
        if model.catalog.revision != config.revision:
            raise RuntimeAssemblyError(
                "launch revision does not match the immutable catalog revision"
            )
        artifact = open_verified_artifact(
            config.artifact_root,
            model,
            expected_content_id=config.artifact_content_id,
        )
        target = _device_target(config.backend)
        model.profile.require_backend(config.backend, target=target)
        resident = _load_family(model, artifact, config.backend)
    except (RuntimeModelResolutionError, RuntimeProfileError, ArtifactError) as error:
        raise RuntimeAssemblyError(str(error)) from error

    executor = TorchDecisionRowExecutor(resident, model.profile)
    descriptor = ModelDescriptor(
        name=model.catalog.model_id,
        description=f"Decision {model.profile.family} model",
        release_date="unknown",
    )
    backend = PhysicalBatchBackend(
        descriptor,
        executor,
        physical_batch_size=config.max_batch,
        max_pending_rows=MAX_PENDING_ROWS,
    )
    scheduler = ModelScheduler(
        (descriptor.name,),
        max_concurrency=config.max_concurrency,
        max_queue=config.max_queue,
    )
    return AssembledRuntime(backend=backend, scheduler=scheduler)


def _validate_config(config: RuntimeLaunchConfig) -> None:
    if not isinstance(config, RuntimeLaunchConfig):
        raise RuntimeAssemblyError("Decision launch configuration is invalid")
    if config.backend not in {"cpu", "rocm", "cuda"}:
        raise RuntimeAssemblyError("Decision backend is unsupported")
    for name, value, minimum, maximum in (
        ("port", config.port, 1, 65535),
        ("max_batch", config.max_batch, 1, MAX_PENDING_ROWS),
        ("max_concurrency", config.max_concurrency, 1, None),
        ("max_queue", config.max_queue, 0, None),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            raise RuntimeAssemblyError(f"Decision {name} is outside its limit")


def _device_target(backend: str) -> str | None:
    """Bind qualified GPU profiles to the device selected by family loaders."""

    if backend == "cpu":
        return None
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeAssemblyError("Decision GPU backend has no available device")
    if backend == "rocm":
        if torch.version.hip is None:
            raise RuntimeAssemblyError("Decision ROCm backend requires HIP Torch")
        target = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        target = target.split(":", 1)[0]
    else:
        if torch.version.hip is not None:
            raise RuntimeAssemblyError("Decision CUDA backend requires CUDA Torch")
        major, minor = torch.cuda.get_device_capability(0)
        target = f"sm{major}{minor}"
    if not target:
        raise RuntimeAssemblyError("Decision GPU target could not be identified")
    return target


def _load_family(
    model: ResolvedRuntimeModel,
    artifact: VerifiedArtifact,
    backend: Literal["cpu", "rocm", "cuda"],
):
    profile = model.profile
    manifest = profile.artifact.manifest
    if profile.family == "vela":
        from .vela_torch import VelaTorchRuntime

        if manifest.path != "native/MANIFEST.json":
            raise RuntimeAssemblyError("Vela release manifest layout is unsupported")
        return VelaTorchRuntime.load(
            artifact.data_root,
            max_length=profile.max_input_tokens,
            backend=backend,
            expected_manifest_sha256=manifest.sha256,
        )
    if profile.family == "qwen3.5":
        from .qwen35_torch import Qwen35TorchRuntime

        if manifest.path not in {"MODEL_MANIFEST.json", "bundle-manifest.json"}:
            raise RuntimeAssemblyError("Qwen release manifest layout is unsupported")
        binder = None
        if backend == "rocm":
            from .qwen35_rocm_binder import create_qwen_rocm_profile_binder

            binder = create_qwen_rocm_profile_binder()
        if profile.temperature is None:
            raise RuntimeAssemblyError("Qwen release has no calibration temperature")
        return Qwen35TorchRuntime.load(
            artifact.data_root,
            temperature=profile.temperature,
            max_length=profile.max_input_tokens,
            backend=backend,
            rocm_profile_binder=binder,
            expected_manifest_sha256=(
                manifest.sha256 if manifest.path == "MODEL_MANIFEST.json" else None
            ),
        )
    raise RuntimeAssemblyError("Decision model family has no owned loader")
