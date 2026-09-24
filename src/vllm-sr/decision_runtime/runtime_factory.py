"""Assemble one pinned Decision artifact into a resident HTTP backend."""

from __future__ import annotations

import importlib
import json
import math
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
from .metrics import RuntimeMetrics
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
        artifact = open_verified_artifact(
            config.artifact_root,
            model,
            expected_content_id=config.artifact_content_id,
        )
        target = _device_target(config.backend)
        require_runtime_backend(
            model.catalog, model.profile, config.backend, target=target
        )
        resident = _load_family(
            model, artifact, config.backend, physical_batch_size=config.max_batch
        )
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
        metrics=metrics,
    )
    scheduler = ModelScheduler(
        (descriptor.name,),
        max_concurrency=config.max_concurrency,
        max_queue=config.max_queue,
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
):
    profile = model.profile
    manifest = getattr(artifact, "manifest", None) or profile.artifact.manifest
    if profile.family == "vela":
        from .vela_torch import VelaTorchRuntime  # noqa: PLC0415

        if manifest.path != "native/MANIFEST.json":
            raise RuntimeAssemblyError("Vela release manifest layout is unsupported")
        return VelaTorchRuntime.load(
            artifact.data_root,
            max_length=profile.max_input_tokens,
            backend=backend,
            expected_manifest_sha256=manifest.sha256,
        )
    if profile.family == "qwen3.5":
        from .qwen35_torch import Qwen35TorchRuntime  # noqa: PLC0415

        policy = profile.qwen_gated_delta_kernel
        if policy is None:
            raise RuntimeAssemblyError(
                "Qwen runtime has no GatedDeltaNet kernel policy"
            )
        if manifest.path not in {"MODEL_MANIFEST.json", "bundle-manifest.json"}:
            raise RuntimeAssemblyError("Qwen release manifest layout is unsupported")
        binder = None
        if backend == "rocm" and policy == "accelerated":
            from .qwen35_rocm_binder import (  # noqa: PLC0415
                create_qwen_rocm_profile_binder,
            )

            binder = create_qwen_rocm_profile_binder()
        temperature = _qwen_temperature(artifact, fallback=profile.temperature)
        return Qwen35TorchRuntime.load(
            artifact.data_root,
            temperature=temperature,
            max_length=profile.max_input_tokens,
            backend=backend,
            gated_delta_kernel_policy=policy,
            physical_batch_size=physical_batch_size,
            rocm_profile_binder=binder,
            expected_manifest_sha256=(
                manifest.sha256 if manifest.path == "MODEL_MANIFEST.json" else None
            ),
        )
    raise RuntimeAssemblyError("Decision model family has no owned loader")


def _qwen_temperature(artifact: VerifiedArtifact, *, fallback: float | None) -> float:
    """Read calibration from the verified snapshot, not its template revision."""

    # Unit assemblers may inject a synthetic artifact without a receipt. Real
    # materializations always include manifest identity and selected file data.
    if getattr(artifact, "manifest", None) is None:
        if fallback is None:
            raise RuntimeAssemblyError("Qwen release has no calibration temperature")
        return fallback

    selected = {item.manifest_path for item in artifact.files}
    source = "temperature.json" if "temperature.json" in selected else "runtime.json"
    try:
        metadata = json.loads((artifact.data_root / source).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeAssemblyError("Qwen calibration metadata is invalid") from error
    temperature = metadata.get("temperature") if isinstance(metadata, dict) else None
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
    ):
        raise RuntimeAssemblyError("Qwen calibration temperature is invalid")
    return float(temperature)
