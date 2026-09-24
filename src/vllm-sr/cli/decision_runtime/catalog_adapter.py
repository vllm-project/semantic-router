"""Integrated catalog resolver for ``vllm-sr decision serve``.

The CLI process resolves the exact catalog revision, verifies and materializes
data-only model artifacts on the host, and passes a read-only content-addressed
tree to one model runtime.  Runtime images never receive Hugging Face
credentials and never execute repository code.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from decision_runtime.artifacts import (
    ArtifactError,
    ArtifactResolver,
    HfHubArtifactFetcher,
    VerifiedArtifact,
)
from decision_runtime.backend_capabilities import require_runtime_backend
from decision_runtime.catalog_adapter import (
    ResolvedRuntimeModel,
    RuntimeModelResolutionError,
    resolve_decision_runtime_model,
)
from decision_runtime.qwen35_torch import EXPERIMENTAL_SOL_GRAPH_MODEL_ID
from decision_runtime.scheduler import DEFAULT_MAX_CONCURRENCY

from cli.decision_runtime.catalog import (
    DecisionCatalogError,
    DecisionRuntimeMount,
    DecisionRuntimeRequest,
    ResolvedDecisionRuntime,
)
from cli.decision_runtime.image_lock import (
    DecisionImageLockError,
    load_decision_image_lock,
)
from cli.decision_runtime.image_reference import (
    ImmutableImageReferenceError,
    validate_decision_image_reference,
    validate_immutable_image_reference,
)

_ARTIFACT_TARGET = "/opt/vllm-sr/decision-artifact"
_CONTAINER_PORT = 8000
# Large multi-state requests wait for row credits while the physical backend
# drains B8 cohorts; the admission queue is independently configurable.
_DEFAULT_DECISION_MAX_QUEUE = 32
_EXPERIMENTAL_GRAPH_BATCH_SIZE = 8
_SHA256_HEX_LENGTH = 64
_SUPPORTED_CONTAINER_BACKENDS = frozenset({"rocm", "cuda", "cpu"})
_FAMILY_PYTHON = MappingProxyType(
    {
        "vela": "/opt/vllm-sr/venvs/vela/bin/python",
        "qwen3.5": "/opt/vllm-sr/venvs/qwen35/bin/python",
    }
)


class DecisionArtifactMaterializer(Protocol):
    """Host-side seam for resolving one immutable data-only artifact tree."""

    def materialize(self, model: ResolvedRuntimeModel) -> VerifiedArtifact:
        """Return a verified content-addressed artifact for ``model``."""


@dataclass(frozen=True, slots=True)
class IntegratedDecisionCatalogResolver:
    """Resolve one public ``decision serve`` request into an immutable OCI launch."""

    artifacts: DecisionArtifactMaterializer
    images: Mapping[str, str] | None = None
    detect_backend: Callable[[], str] = field(default=lambda: detect_backend)

    def __post_init__(self) -> None:
        if self.images is None:
            return
        try:
            images = dict(self.images.items())
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError(
                "Decision runtime image inventory must be a mapping"
            ) from error
        object.__setattr__(self, "images", MappingProxyType(images))

    def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
        """Resolve, verify, and materialize all launch inputs before mutation."""

        if not isinstance(request, DecisionRuntimeRequest):
            raise DecisionCatalogError("Decision runtime request is invalid")
        if type(request.experimental_qwen_rocm_graph_b8) is not bool:
            raise DecisionCatalogError(
                "experimental Qwen graph opt-in must be a boolean"
            )
        backend = (
            self.detect_backend() if request.backend == "auto" else request.backend
        )
        if backend not in _SUPPORTED_CONTAINER_BACKENDS:
            raise DecisionCatalogError(
                "Decision runtime backend must resolve to rocm, cuda, or cpu; "
                "native MLX integration is not available in this build"
            )

        try:
            model = resolve_decision_runtime_model(
                request.model, revision=request.revision, backend=backend
            )
        except RuntimeModelResolutionError as error:
            raise DecisionCatalogError(str(error)) from error

        qualification = require_runtime_backend(model.catalog, model.profile, backend)
        dtype = qualification.backbone_dtype
        if dtype is None:  # pragma: no cover - qualified profiles require a dtype
            raise DecisionCatalogError("qualified Decision backend has no dtype")
        if request.dtype is not None and request.dtype != dtype:
            raise DecisionCatalogError(
                f"revision {model.catalog.revision} requires backbone dtype {dtype!r}"
            )
        image = request.image
        if image is None:
            try:
                inventory = (
                    self.images
                    if self.images is not None
                    else load_decision_image_lock().images
                )
            except DecisionImageLockError as error:
                raise DecisionCatalogError(str(error)) from error
            image = inventory.get(backend)
        if image is None:
            raise DecisionCatalogError(
                f"This CLI build has no default Decision image for {backend!r}. "
                "Install a release with this backend, or use --image for local development."
            )
        try:
            image = (
                validate_decision_image_reference(image)
                if request.image is not None
                else validate_immutable_image_reference(image)
            )
        except ImmutableImageReferenceError as error:
            raise DecisionCatalogError(
                f"Decision runtime image for {backend!r} is invalid: {error}"
            ) from error

        max_batch = request.max_batch or model.profile.physical_batch_size
        max_concurrency = request.max_concurrency or DEFAULT_MAX_CONCURRENCY
        max_queue = (
            _DEFAULT_DECISION_MAX_QUEUE
            if request.max_queue is None
            else request.max_queue
        )
        _validate_resolved_limits(max_batch, max_concurrency, max_queue)
        if request.experimental_qwen_rocm_graph_b8 and (
            model.catalog.model_id != EXPERIMENTAL_SOL_GRAPH_MODEL_ID
            or backend != "rocm"
            or max_batch != _EXPERIMENTAL_GRAPH_BATCH_SIZE
        ):
            raise DecisionCatalogError(
                "experimental Qwen ROCm graph requires canonical Sol at B8"
            )

        try:
            artifact = self.artifacts.materialize(model)
        except ArtifactError as error:
            raise DecisionCatalogError(
                f"Decision model artifact verification failed: {error}"
            ) from error
        if (
            artifact.repository_id != model.repository_id
            or artifact.revision != model.catalog.revision
        ):
            raise DecisionCatalogError(
                "Decision artifact resolver returned a different model revision"
            )
        artifact_root = _canonical_artifact_root(artifact)
        python = _FAMILY_PYTHON.get(model.profile.family)
        if python is None:  # pragma: no cover - guarded by the profile parser
            raise DecisionCatalogError(
                "Decision runtime family has no image entrypoint"
            )

        command = (
            python,
            "-m",
            "decision_runtime.entrypoint",
            "--model",
            model.catalog.model_id,
            "--revision",
            model.catalog.revision,
            "--backend",
            backend,
            "--artifact-root",
            _ARTIFACT_TARGET,
            "--artifact-content-id",
            artifact.content_id,
            "--host",
            "0.0.0.0",
            "--port",
            str(_CONTAINER_PORT),
            "--max-batch",
            str(max_batch),
            "--max-concurrency",
            str(max_concurrency),
            "--max-queue",
            str(max_queue),
        )
        if request.experimental_qwen_rocm_graph_b8:
            command += ("--experimental-qwen-rocm-graph-b8",)
        return ResolvedDecisionRuntime(
            canonical_model=model.catalog.model_id,
            revision=model.catalog.revision,
            backend=backend,
            dtype=dtype,
            image=image,
            artifact_digest=f"sha256:{artifact.content_id}",
            max_batch=max_batch,
            max_concurrency=max_concurrency,
            max_queue=max_queue,
            command=command,
            environment={"TOKENIZERS_PARALLELISM": "false"},
            mounts=(
                DecisionRuntimeMount(
                    source=str(artifact_root),
                    target=_ARTIFACT_TARGET,
                ),
            ),
            container_port=_CONTAINER_PORT,
        )


def detect_backend() -> str:
    """Detect one unambiguous supported accelerator without importing frameworks."""

    if sys.platform == "darwin":
        raise DecisionCatalogError(
            "native MLX launch support is not available in this build; "
            "do not emulate it through an OCI runtime"
        )
    rocm = Path("/dev/kfd").exists()
    cuda = Path("/dev/nvidiactl").exists()
    if rocm and cuda:
        raise DecisionCatalogError(
            "both ROCm and CUDA devices are visible; select --backend explicitly"
        )
    if rocm:
        return "rocm"
    if cuda:
        return "cuda"
    return "cpu"


def default_artifact_cache_root() -> Path:
    """Return an absolute user cache root without accepting ambiguous paths."""

    configured = os.getenv("XDG_CACHE_HOME", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    if not root.is_absolute():
        raise DecisionCatalogError("XDG_CACHE_HOME must be an absolute path")
    return root / "vllm-sr" / "decision-runtime" / "artifacts"


def get_catalog_resolver() -> IntegratedDecisionCatalogResolver:
    """Build the production host-side resolver used by ``vllm-sr decision serve``."""

    return IntegratedDecisionCatalogResolver(
        artifacts=ArtifactResolver(
            fetcher=HfHubArtifactFetcher(),
            cache_root=default_artifact_cache_root(),
        )
    )


def _canonical_artifact_root(artifact: VerifiedArtifact) -> Path:
    if not isinstance(artifact, VerifiedArtifact):
        raise DecisionCatalogError("artifact resolver returned an invalid receipt")
    if len(artifact.content_id) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in artifact.content_id
    ):
        raise DecisionCatalogError("artifact resolver returned an invalid content ID")
    requested = artifact.root
    if requested.is_symlink():
        raise DecisionCatalogError("artifact resolver returned a symlink root")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise DecisionCatalogError(
            "artifact resolver returned an unavailable root"
        ) from error
    if not resolved.is_dir() or resolved != requested:
        raise DecisionCatalogError(
            "artifact resolver returned a non-canonical artifact root"
        )
    return resolved


def _validate_resolved_limits(
    max_batch: object, max_concurrency: object, max_queue: object
) -> None:
    for label, value in (
        ("max batch", max_batch),
        ("max concurrency", max_concurrency),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DecisionCatalogError(f"resolved {label} must be positive")
    if isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue < 0:
        raise DecisionCatalogError("resolved max queue must be non-negative")
