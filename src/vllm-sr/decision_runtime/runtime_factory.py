"""Assemble one pinned Decision artifact into a resident HTTP backend."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .artifacts import ArtifactError, VerifiedArtifact, open_verified_artifact
from .backend import ModelDescriptor
from .backend_capabilities import require_runtime_backend
from .catalog_adapter import (
    ResolvedRuntimeModel,
    RuntimeModelResolutionError,
    resolve_decision_runtime_model,
)
from .families.base import FamilyLoadError
from .family_registry import family_adapter
from .metrics import RuntimeMetrics
from .physical_batching import PhysicalBatchBackend
from .row_executor import TorchDecisionRowExecutor
from .runtime_limits import MAX_PENDING_ROWS
from .runtime_profile import RuntimeProfileError
from .scheduler import ModelScheduler


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
    artifact_provenance: dict[str, str] | None = None


def assemble_runtime(
    config: RuntimeLaunchConfig, *, metrics: RuntimeMetrics | None = None
) -> AssembledRuntime:
    """Verify identity and bytes before loading any model framework."""

    _validate_config(config)
    try:
        model = resolve_decision_runtime_model(
            config.model, revision=config.revision, backend=config.backend
        )
        model.profile.validate_physical_batch_size(config.backend, config.max_batch)
        artifact = open_verified_artifact(
            config.artifact_root,
            model,
            expected_content_id=config.artifact_content_id,
        )
        target = _device_target(config.backend)
        require_runtime_backend(
            model.catalog, model.profile, config.backend, target=target
        )
        load_options = {"physical_batch_size": config.max_batch}
        if metrics is not None and model.profile.use_short_b8_graph(
            config.backend, config.max_batch
        ):
            load_options["graph_event_recorder"] = lambda event: (
                metrics.record_qwen_rocm_graph_event(model.catalog.model_id, event)
            )
        resident = _load_family(model, artifact, config.backend, **load_options)
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
        max_rows_per_job_turn=model.profile.rows_per_job_turn(
            config.backend, config.max_batch
        ),
        metrics=metrics,
    )
    scheduler = ModelScheduler(
        (descriptor.name,),
        max_concurrency=config.max_concurrency,
        max_queue=config.max_queue,
        max_active_rows=backend.max_pending_rows,
    )
    manifest = getattr(artifact, "manifest", None)
    provenance = (
        {
            "model": model.catalog.model_id,
            "revision": artifact.revision,
            "manifest_sha256": manifest.sha256,
            "content_sha256": artifact.content_id,
        }
        if manifest is not None
        else None
    )
    return AssembledRuntime(
        backend=backend, scheduler=scheduler, artifact_provenance=provenance
    )


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
    *,
    physical_batch_size: int = 8,
    graph_event_recorder: Callable[[str], None] | None = None,
):
    try:
        adapter = family_adapter(model.profile.family)
    except ValueError as error:
        raise RuntimeAssemblyError(str(error)) from error
    try:
        return adapter.load(
            artifact,
            model.profile,
            backend,
            physical_batch_size=physical_batch_size,
            graph_event_recorder=graph_event_recorder,
        )
    except FamilyLoadError as error:
        raise RuntimeAssemblyError(str(error)) from error
