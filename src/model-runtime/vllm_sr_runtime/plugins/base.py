"""Plugin interfaces: model family, engine, accelerator and numerics profile.

A family owns a model format's task contract (package format, rendering,
readout, answers, limits) and declares the API surfaces it serves. An engine
executes a backbone described by a ``ModelSpec`` on an accelerator and returns
gathered hidden rows (decoders), hidden states at layer exits (encoders) or
named graph outputs. An accelerator describes a device class and supplies
kernels. A profile is a numerics and scheduling policy. Built-in plugins
register through the same entry points as third-party ones
(``plugins/registry.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    import torch

    from ..accel.kernels import KernelSet

# The API surfaces a model may serve (``/v1/<surface>``).
SURFACES = ("decisions", "classify", "embeddings", "rerank")

# A job whose deadline passed in the queue resolves to this instead of results.
DEADLINE = object()


class UnsupportedSurfaceError(Exception):
    """The model does not serve the requested surface (HTTP 422)."""

    def __init__(self, surface: str, model: str | None = None):
        where = f" {model}" if model else ""
        super().__init__(f"model{where} does not serve /v1/{surface}")
        self.surface = surface


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
    """Resolution and policy settings a family needs while verifying a package.

    ``model_options`` are the served model's family options (``--models``
    file ``options``), for example a pinned pair-scorer exit; families ignore
    keys they do not define and reject values they cannot honour.
    """

    cache_dir: str | Path | None = None
    offline: bool = False
    base_path: str | Path | None = None
    accept_licences: tuple[str, ...] = ()
    model_options: Mapping[str, Any] = field(default_factory=dict)


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
    BF16 autocast multiplies with anyway. ``gpu_weights`` holds every backbone
    parameter in that dtype on GPUs, so the hidden-state stream follows it
    (released runtimes that load the backbone in BF16); CPU keeps ``weights``.
    ``reduced_gpu`` / ``reduced_cpu`` consent to a reduced-precision copy of
    the backbone's linear layers when the configured profile asks for one
    (``EngineOptions.reduced_precision``): ``"bfloat16"`` on GPUs, ``"int8"``
    (dynamic) or ``"bfloat16"`` on CPUs; None keeps FP32. A family consents only
    where its records show at least 99% label agreement with the exact path
    (embeddings: cosine of at least 0.999).
    """

    weights: str = "float32"
    autocast: str | None = "bfloat16"
    head: str = "float32"
    bf16_resident: bool = True
    gpu_weights: str | None = None
    reduced_gpu: str | None = None
    reduced_cpu: str | None = None


@dataclass(frozen=True)
class LoRASpec:
    """An unmerged PEFT LoRA adapter over a pinned base backbone."""

    adapter_config: dict[str, Any]
    weight_files: tuple[Path, ...]
    rank: int
    alpha: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class BranchSpec:
    """Another layer stack of a branched encoder over the backbone's embedding.

    Its weights are named ``<layers>.<i>.*`` and ``<final_norm>.*`` in ``weight_files``.
    """

    weight_files: tuple[Path, ...]
    layers: str
    final_norm: str


@dataclass(frozen=True)
class BackboneSpec:
    """A backbone architecture and the files that hold its weights.

    ``branches`` names extra layer stacks that share the embedding; an encoder
    batch selects one with ``EncoderBatch.branch``.
    """

    model_type: str
    config: dict[str, Any]
    weight_files: tuple[Path, ...]
    weight_prefix: str = ""
    lora: LoRASpec | None = None
    branches: Mapping[str, BranchSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Everything an engine needs to build and run a backbone.

    ``graphs`` names ONNX graphs a package ships (``{"default": path}``), for
    engines that run a graph instead of building the backbone; ``encoder``
    marks a bidirectional encoder whose readout needs hidden states.
    ``kernel_variants`` maps a kernel slot to the variant the model's released
    runtime ran (``KernelSet.use_variants``); a slot without that variant on
    the device runs its reference.
    """

    name: str
    backbone: BackboneSpec
    dtype: DtypePolicy
    max_input_tokens: int
    graphs: Mapping[str, Path] = field(default_factory=dict)
    encoder: bool = False
    kernel_variants: Mapping[str, str] = field(default_factory=dict)


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
    """Execution switches a profile may change. Defaults are the exact path.

    ``reduced_precision`` loads, next to the exact weights, the reduced copy the
    model's ``DtypePolicy`` consents to on its device.
    """

    graphs: bool = True
    fused_kernels: bool = True
    exact_kernels_only: bool = True
    merge_lora: bool = False
    reduced_precision: bool = False
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


@dataclass
class EncoderBatch:
    """Rows for a bidirectional encoder: padded, or packed back to back.

    Padded: ``input_ids`` and ``attention_mask`` are ``[rows, tokens]`` and
    hidden states come back as ``[rows, tokens, hidden]``. Packed: ``lengths``
    (host-known) splits ``input_ids`` ``[N]`` into rows, ``attention_mask`` is
    unused, and hidden states come back as ``[N, hidden]``, so no position is
    padding. ``layers`` are the hidden-state exits the readout needs: 1-based layer
    indices, 0 for the embedding output, empty for the last layer only. An
    intermediate exit is the raw residual stream and the last layer is
    final-normalized (the Transformers ``hidden_states`` convention), unless
    ``normalize_exits`` asks for the final norm at every exit. One forward
    serves every requested exit. ``graph`` names the ``ModelSpec.graphs``
    entry an engine that runs graphs executes, ``graph_inputs`` are extra named
    inputs for it, and ``outputs`` the graph outputs the readout reads.
    ``branch`` runs one of the backbone's branches (``BackboneSpec.branches``)
    instead of its own layer stack. ``reduced`` runs the reduced-precision copy
    when the engine loaded one (approximate batches only), else the exact weights.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    layers: tuple[int, ...] = ()
    normalize_exits: bool = False
    graph: str = "default"
    graph_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    lengths: list[int] | None = None
    branch: str | None = None
    reduced: bool = False


@dataclass
class EncoderOutput:
    """Hidden states by layer exit ([batch, tokens, hidden]) and named graph outputs."""

    hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class TreeBatch:
    """Causal prefixes computed once, and blocks that each continue from one of them.

    Block ``i`` attends to prefix ``owners[i]`` and to itself (causally), with
    positions that continue after the prefix, so its states equal those of the
    sequence prefix + block. ``layout`` sets how an engine lays the tokens out,
    which changes rounding only: ``packed`` runs each prefix and its blocks back
    to back in one row (the fewest tokens); ``rows`` runs left-padded prefix rows
    and right-padded blocks as two tensors (the tensor shapes and operations of
    the Vela 2.0 packages' engine).
    """

    prefixes: list[list[int]]
    blocks: list[list[int]]
    owners: list[int]
    layout: Literal["packed", "rows"] = "packed"


@dataclass
class TreeOutput:
    """Final hidden states of every block's tokens, ``[blocks, width, hidden]`` (rows padded at the end)."""

    hidden: torch.Tensor


class EngineModel(ABC):
    """A backbone loaded on one device.

    ``hidden_states`` says whether ``encode`` returns hidden states at any
    requested exit (an engine that builds the backbone) or the named outputs
    of the graph a batch names (an engine that runs graphs).
    """

    device: torch.device
    device_info: DeviceInfo
    hidden_states: ClassVar[bool] = True

    @abstractmethod
    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        """Run the backbone and gather the requested rows (on the device)."""

    def encode(self, batch: EncoderBatch) -> EncoderOutput:
        """Run an encoder and return hidden states at the requested exits or graph outputs."""
        raise NotImplementedError(f"{type(self).__name__} has no encoder forward")

    def tree(self, batch: TreeBatch) -> TreeOutput:
        """Run a prefix once and every block from it (decoder backbones with a tree forward)."""
        raise NotImplementedError(f"{type(self).__name__} has no tree forward")

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

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        """Capability descriptor listed in ``/v1/models``: architectures, outputs, devices."""
        return {}

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

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        """Capability descriptor listed in ``/v1/models``; per-device facts come from ``capabilities``."""
        return {"validated": cls.validated}

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

    def execute(self, device: DeviceInfo, work: Callable[[], Any]) -> Any:
        """Run device work (loading, forwards, readouts); inline unless the device needs one thread."""
        return work()


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
class HeadInfo:
    """A fixed head served on ``/v1/classify``.

    ``kind``: ``sequence`` (softmax distribution), ``scores`` (independent
    sigmoid per label) or ``token`` (spans). ``inputs``: ``text``, ``pair``
    and / or ``grounded``. ``thresholds`` is a packaged operating point (one
    threshold per label); ``window`` the default ``(tokens, overlap)`` for
    ``overflow: window``; ``reduction`` how windows combine when the head
    declares one (``max``, ``span_union``).
    """

    name: str
    kind: str
    labels: tuple[str, ...]
    inputs: tuple[str, ...] = ("text",)
    default_threshold: float | None = None
    thresholds: tuple[float, ...] | None = None
    overflow: str = "reject"
    window: tuple[int, int] | None = None
    reduction: str | None = None


@dataclass(frozen=True)
class EmbeddingInfo:
    """What ``/v1/embeddings`` accepts: dimensions and layer exits (the last is the default)."""

    dimensions: tuple[int, ...]
    layers: tuple[int, ...]
    modalities: tuple[str, ...] = ("text",)
    normalized: bool = True
    pooling: str = "mean"
    input_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class RerankInfo:
    """Pair-scorer exits ``(layer, dimension)`` served on ``/v1/rerank``; scores are raw logits."""

    exits: tuple[tuple[int, int], ...]
    default: tuple[int, int]


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
    heads: tuple[HeadInfo, ...] = ()
    embedding: EmbeddingInfo | None = None
    rerank: RerankInfo | None = None
    presets: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceRequest:
    """A validated request to one surface of one model.

    ``body`` is the JSON body; the runtime has checked its top-level fields
    and parsed the generic options (deadline, profile, return_meta). The
    family validates everything else in ``plan_surface``.
    """

    surface: str
    body: dict[str, Any]
    deadline: float | None
    profile: str
    return_meta: bool
    received: float

    @property
    def options(self) -> dict[str, Any]:
        options = self.body.get("options")
        return options if isinstance(options, dict) else {}


@dataclass
class SurfacePlan:
    """A rendered request: work items for the scheduler and private assembly state.

    Every item has ``ids`` (its token IDs), which bound admission and batches;
    ``run`` receives the items and ``finish_surface`` their results in order.
    """

    surface: str
    items: list[Any]
    input_tokens: int
    state: Any = None


class LoadedModel(ABC):
    """A family's model bound to an engine model.

    The runtime calls ``plan_surface`` on the request thread, the scheduler
    calls ``run`` on the model's worker with micro-batches of the plan's
    items, and ``finish_surface`` turns the results into the surface's
    response body: a list in item order, or ``DEADLINE`` when the plan
    expired in the queue; with the result cache an item that expired while
    others were cached is ``DEADLINE`` in the list. Results are shared with
    the cache, so ``finish_surface`` must not mutate them. An item that sets
    ``cache_key`` (a content hash of everything its result depends on) may be
    answered from the per-model result cache instead of a forward. The default
    surface methods serve ``/v1/decisions`` through ``plan`` and ``answer``,
    the Phase 1 decision interface, which encoder families need not implement.

    ``fuse_bundled_jobs`` lets the ``exact`` profile run the jobs of one
    bundle as one batch, so a family can compute each distinct input once for
    several heads; decision families keep it off because their released
    numerics batch one request at a time.
    """

    info: ModelInfo
    engine_model: EngineModel
    fuse_bundled_jobs: ClassVar[bool] = False

    def plan(self, state: Any, questions: dict[str, Any]) -> RequestPlan:
        """Validate and render every question; failures become per-question errors."""
        raise UnsupportedSurfaceError("decisions", self.info.id)

    @abstractmethod
    def run(self, items: list[Any], shared_prefix: int = 0) -> list[Any]:
        """One forward over ``items`` in order; per item its readout (decisions: option logits).

        ``shared_prefix`` > 0 runs the items as one shared-context tree (see ``ForwardBatch``).
        """

    def run_approximate(self, items: list[Any]) -> list[Any]:
        """``run`` for a batch of an approximate profile, where a family may trade exactness for speed."""
        return self.run(items)

    def answer(self, item: RenderedItem, logits: list[float] | None) -> dict[str, Any]:
        """The API answer for one decision item."""
        raise UnsupportedSurfaceError("decisions", self.info.id)

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan:
        """Validate and render a request; per-item failures stay in the plan."""
        if surface != "decisions" or "decisions" not in self.info.surfaces:
            raise UnsupportedSurfaceError(surface, self.info.id)
        plan = self.plan(request.body["state"], request.body["questions"])
        return SurfacePlan("decisions", list(plan.items), plan.input_tokens, plan)

    def finish_surface(self, plan: SurfacePlan, results: Any) -> dict[str, Any]:
        """The response body without ``model``, ``usage`` and ``meta`` (the runtime adds those)."""
        from ..errors import DEADLINE_EXCEEDED

        request_plan: RequestPlan = plan.state
        answered: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(request_plan.items):
            if results is DEADLINE:
                answered[item.question_id] = {
                    "type": item.task_type,
                    "error": DEADLINE_EXCEEDED,
                }
            else:
                answered[item.question_id] = self.answer(item, results[index])
        return {
            "answers": {
                question_id: request_plan.errors.get(question_id)
                or answered[question_id]
                for question_id in request_plan.question_ids
            }
        }

    def golden_values(self, surface: str, response: dict[str, Any]) -> dict[str, float]:
        """Comparable numbers of a golden response (non-decision surfaces), by stable key."""
        from ..supervision.readiness import flatten

        return flatten(surface, response)

    def forward_token_budget(self) -> int | None:
        """Most padded tokens one forward may hold; None when nothing limits it."""
        return None

    def shared_context(self, items: list[Any], token_budget: int | None) -> int | None:
        """How a job's items share context on the shared-context path.

        A positive value runs them through ``run(items, shared_prefix=value)``,
        0 runs them exactly, and None lets the profile find the common token
        prefix of decision items itself.
        """
        return None

    def exact_batches(self, items: list[Any]) -> list[list[int]] | None:
        """How the released runtime splits one request's items into forwards (indices into ``items``).

        None (the default) is one padded batch, split only by the forward
        token budget. A family whose released runtime batches otherwise (for
        example in fixed physical batches) returns its split, which ``exact``
        and the exact fallback of the other profiles then run.
        """
        return None

    def close(self) -> None:
        self.engine_model.close()


class ModelFamily(ABC):
    name: ClassVar[str]
    surfaces: ClassVar[frozenset[str]]

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        """Capability descriptor listed in ``/v1/models``: surfaces and package formats."""
        return {"surfaces": sorted(cls.surfaces)}

    def __init__(self, options: RegistryOptions | None = None):
        self.options = options or RegistryOptions()

    @abstractmethod
    def detect(self, package: PackageRef) -> bool:
        """Cheap ownership test; reads the package pointer only."""

    def fetch(self, package: PackageRef) -> PackageRef:
        """Download the files this family loads when the resolver could not know them.

        The resolver fetches a built-in entry's pinned files or a manifest's
        files; a Hub package with neither arrives with its pointer files only,
        and a family that serves such packages downloads its inventory here
        (``registry.resolve.fetch``). Verification still follows.
        """
        return package

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

    def kernel_choices(
        self, package: VerifiedPackage, device: DeviceInfo
    ) -> dict[str, Any]:
        """Recorded autotuned-kernel configurations for this model on the device's class, if any."""
        from ..registry import builtin

        known = builtin.by_identity(package.model_sha256)
        if known is None or not device.arch:
            return {}
        return known.kernel_choices.get(f"{device.accelerator}:{device.arch}", {})


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One request's rendered items waiting for execution.

    Jobs submitted together for one model (a bundle's tasks) share a
    ``group``; ``exact`` may run a group as one batch when the model allows it
    (``LoadedModel.fuse_bundled_jobs``).
    """

    items: list[Any]
    deadline: float | None
    enqueued: float
    profile: str
    payload: Any = None
    group: int | None = None


@dataclass
class Batch:
    """Items from one or more jobs that run as one forward, in order.

    ``shared_prefix`` > 0 (shared-context profile): the items' common prefix length, run once.
    ``exact`` False marks a batch an approximate profile formed (``LoadedModel.run_approximate``).
    """

    parts: list[tuple[Job, list[int]]]
    shared_prefix: int = 0
    exact: bool = True

    def items(self) -> list[RenderedItem]:
        return [job.items[index] for job, indices in self.parts for index in indices]


class Profile(ABC):
    """How queued jobs become forwards.

    ``coalesces`` profiles merge jobs of concurrent requests into one batch, so
    the scheduler holds a queue that has any of their jobs for the batching
    window before planning it.
    """

    name: ClassVar[str]
    numerics: ClassVar[Literal["exact", "approximate"]]
    description: ClassVar[str]
    coalesces: ClassVar[bool] = False

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        """Capability descriptor listed in ``/v1/models``."""
        return {"numerics": cls.numerics, "coalesces": cls.coalesces}

    def engine_options(self, base: EngineOptions) -> EngineOptions:
        return base

    def available(self, model: LoadedModel) -> str | None:
        """None when the profile can run on ``model``, else the reason it cannot."""
        return None

    @abstractmethod
    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        """Turn pending jobs into forwards; every item of every job must appear once."""
