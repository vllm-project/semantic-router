"""Plugin interfaces: model family, engine, accelerator and numerics profile.

A family owns a model format's task contract (package format, rendering,
readout, answers, limits). An engine executes a backbone described by a
``ModelSpec`` on an accelerator and returns gathered hidden rows. An
accelerator describes a device class and supplies kernels. A profile is a
numerics and scheduling policy. Built-in plugins register through the same
entry points as third-party ones (``plugins/registry.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    import torch

    from ..accel.kernels import KernelSet

# ---------------------------------------------------------------------------
# Packages and specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackageRef:
    """A model package on local disk, as resolved by the registry."""

    root: Path
    repo_id: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class RegistryOptions:
    """Resolution and policy settings a family needs while verifying a package."""

    cache_dir: str | Path | None = None
    offline: bool = False
    base_path: str | Path | None = None
    accept_licences: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedPackage:
    """A package whose bytes matched its manifest; no model code has run."""

    ref: PackageRef
    family: str
    model_name: str
    manifest: dict[str, Any]
    manifest_sha256: str
    model_sha256: str
    max_input_tokens: int
    licence: str | None
    loaded_parameters: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DtypePolicy:
    """How a backbone holds and computes its tensors on a device.

    ``autocast`` applies only on GPUs; CPU inference runs in ``weights`` dtype.
    ``bf16_resident`` holds BF16-exact Linear weights in BF16 on GPUs, which
    BF16 autocast multiplies with anyway.
    """

    weights: str = "float32"
    autocast: str | None = "bfloat16"
    head: str = "float32"
    bf16_resident: bool = True


@dataclass(frozen=True)
class LoRASpec:
    """An unmerged PEFT LoRA adapter over a pinned base backbone."""

    adapter_config: dict[str, Any]
    weight_files: tuple[Path, ...]
    rank: int
    alpha: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class BackboneSpec:
    """A backbone architecture and the files that hold its weights."""

    model_type: str
    config: dict[str, Any]
    weight_files: tuple[Path, ...]
    weight_prefix: str = ""
    lora: LoRASpec | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Everything an engine needs to build and run a backbone."""

    name: str
    backbone: BackboneSpec
    dtype: DtypePolicy
    max_input_tokens: int


# ---------------------------------------------------------------------------
# Devices and engines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    accelerator: str
    index: int | None
    name: str
    total_memory: int | None = None
    free_memory: int | None = None
    bf16: bool = False
    arch: str | None = None

    @property
    def label(self) -> str:
        return (
            self.accelerator
            if self.index is None
            else f"{self.accelerator}:{self.index}"
        )


@dataclass(frozen=True)
class EngineOptions:
    """Execution switches a profile may change. Defaults are the exact path."""

    graphs: bool = True
    fused_kernels: bool = True
    exact_kernels_only: bool = True
    merge_lora: bool = False
    threads: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardBatch:
    """Padded token rows and the positions whose hidden states the readout needs.

    ``gather`` holds per-row option endpoints padded with 0 to the widest row;
    ``query`` holds one query position per row; ``lengths`` the unpadded lengths.
    ``shared_prefix`` > 0 asks an engine that ``supports_shared_context`` to run the
    rows as one shared-context tree whose first ``shared_prefix`` tokens (common to
    every row, before any endpoint) are computed once.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    gather: torch.Tensor
    query: torch.Tensor
    lengths: list[int]
    shared_prefix: int = 0


@dataclass
class ForwardOutput:
    gathered: torch.Tensor
    query: torch.Tensor


class EngineModel(ABC):
    """A backbone loaded on one device."""

    device: torch.device
    device_info: DeviceInfo

    @abstractmethod
    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        """Run the backbone and gather the requested rows (on the device)."""

    @abstractmethod
    def parameter_count(self) -> int: ...

    def memory_bytes(self) -> int:
        return 0

    def autocast(self) -> AbstractContextManager[Any]:
        from contextlib import nullcontext

        return nullcontext()

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release device memory."""


class Engine(ABC):
    name: ClassVar[str]

    @abstractmethod
    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None:
        """None when the engine can run ``spec`` on ``device``, else the reason it cannot."""

    @abstractmethod
    def load(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> EngineModel: ...


class Accelerator(ABC):
    name: ClassVar[str]
    validated: ClassVar[bool]

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def devices(self) -> list[DeviceInfo]: ...

    @abstractmethod
    def torch_device(self, device: DeviceInfo) -> torch.device: ...

    @abstractmethod
    def kernels(self, device: DeviceInfo) -> KernelSet: ...

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {}

    def autocast(
        self, device: DeviceInfo, dtype: str | None
    ) -> AbstractContextManager[Any]:
        from contextlib import nullcontext

        return nullcontext()

    def synchronize(self, device: DeviceInfo) -> None:  # noqa: B027 - optional hook
        """Wait for queued device work."""


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedItem:
    """One question rendered to model inputs."""

    question_id: str
    task_type: str
    ids: list[int]
    gather: list[int]
    query: int
    keys: list[str]
    descriptions: list[Any]


@dataclass
class RequestPlan:
    """A request after validation and rendering, before execution."""

    question_ids: list[str]
    items: list[RenderedItem]
    errors: dict[str, dict[str, Any]]
    input_tokens: int


@dataclass(frozen=True)
class ModelInfo:
    """What ``/v1/models`` reports about a loaded model."""

    id: str
    family: str
    repo: str | None
    revision: str | None
    model_sha256: str
    manifest_sha256: str
    surfaces: tuple[str, ...]
    question_types: tuple[str, ...]
    limits: dict[str, int]
    licence: str | None
    parameters: int
    dtype: str


class LoadedModel(ABC):
    """A family's model bound to an engine model."""

    info: ModelInfo
    engine_model: EngineModel

    @abstractmethod
    def plan(self, state: Any, questions: dict[str, Any]) -> RequestPlan:
        """Validate and render every question; failures become per-question errors."""

    @abstractmethod
    def run(
        self, items: list[RenderedItem], shared_prefix: int = 0
    ) -> list[list[float] | None]:
        """One forward over ``items`` in order; per item the logits of its options.

        ``shared_prefix`` > 0 runs the items as one shared-context tree (see ``ForwardBatch``).
        """

    @abstractmethod
    def answer(self, item: RenderedItem, logits: list[float] | None) -> dict[str, Any]:
        """The API answer for one item."""

    def forward_token_budget(self) -> int | None:
        """Most padded tokens one forward may hold; None when nothing limits it."""
        return None

    def close(self) -> None:
        self.engine_model.close()


class ModelFamily(ABC):
    name: ClassVar[str]
    surfaces: ClassVar[frozenset[str]]

    def __init__(self, options: RegistryOptions | None = None):
        self.options = options or RegistryOptions()

    @abstractmethod
    def detect(self, package: PackageRef) -> bool:
        """Cheap ownership test; reads the package pointer only."""

    @abstractmethod
    def verify(self, package: PackageRef) -> VerifiedPackage:
        """Check every packaged byte; must not import or execute package code."""

    @abstractmethod
    def describe(self, package: VerifiedPackage) -> ModelSpec: ...

    @abstractmethod
    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel: ...

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        """Golden requests ({state, questions, expected?}) that gate readiness."""
        return []


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One request's rendered items waiting for execution."""

    items: list[RenderedItem]
    deadline: float | None
    enqueued: float
    profile: str
    payload: Any = None


@dataclass
class Batch:
    """Items from one or more jobs that run as one forward, in order.

    ``shared_prefix`` > 0 (shared-context profile): the items' common prefix length, run once.
    """

    parts: list[tuple[Job, list[int]]]
    shared_prefix: int = 0

    def items(self) -> list[RenderedItem]:
        return [job.items[index] for job, indices in self.parts for index in indices]


class Profile(ABC):
    name: ClassVar[str]
    numerics: ClassVar[Literal["exact", "approximate"]]
    description: ClassVar[str]

    def engine_options(self, base: EngineOptions) -> EngineOptions:
        return base

    def available(self, model: LoadedModel) -> str | None:
        """None when the profile can run on ``model``, else the reason it cannot."""
        return None

    @abstractmethod
    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        """Turn pending jobs into forwards; every item of every job must appear once."""
