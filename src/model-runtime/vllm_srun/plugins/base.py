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
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    final,
)

if TYPE_CHECKING:
    import torch

    from ..accel.kernels import KernelSet
    from ..config import ServeConfig

# The API surfaces a model may serve (``/v1/<surface>``).
SURFACES = ("decisions", "classify", "embeddings", "rerank")


@final
class Expired:
    """What a job resolves to, instead of results, when its deadline passed in the queue (``DEADLINE``)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "DEADLINE"


DEADLINE: Final = Expired()

# ``HeadInfo`` fields, as the OpenAPI ``HeadCard`` declares them.
HeadKind = Literal["sequence", "scores", "token"]
Overflow = Literal["reject", "truncate", "window"]
Reduction = Literal["max", "span_union"]


class WorkItem(Protocol):
    """A plan's item as the scheduler and the result cache read it.

    ``ids`` (its token IDs) bound admission and batches. An item may also
    carry ``cost``, the tokens it costs a forward when it has no token IDs
    (images, audio), and ``cache_key``, a content hash of everything its
    result depends on; both are optional attributes.
    """

    @property
    def ids(self) -> Sequence[int]: ...


ItemT = TypeVar("ItemT", bound=WorkItem)
ResultT = TypeVar("ResultT")
ModuleT = TypeVar("ModuleT", bound="torch.nn.Module")
# A plan's results: one per item in order, ``DEADLINE`` for an item that
# expired while others were cached, or ``DEADLINE`` for the whole plan.
Results = list[ResultT | Expired] | Expired


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
    (released runtimes that load the backbone in BF16); CPU keeps ``weights``
    unless ``cpu_weights`` names the dtype a released runtime held there too.
    ``reduced_gpu`` / ``reduced_cpu`` consent to a reduced-precision copy of
    the backbone's linear layers when the configured profile asks for one
    (``EngineOptions.reduced_precision``): ``"bfloat16"`` on GPUs; on CPUs
    ``"bfloat16"`` (only where ``DeviceInfo.bf16``), ``"float32-packed"``
    (oneDNN's pre-packed FP32 linear) or ``"int8"`` (dynamic); None keeps the
    exact weights. A family consents only
    where its records show at least 99% label agreement with the exact path
    (embeddings: cosine of at least 0.999). ``approximate_kernels`` consents,
    on the same evidence per question type, to the accelerator's approximate
    kernels when the profile allows them (``EngineOptions.exact_kernels_only``
    off).
    """

    weights: str = "float32"
    autocast: str | None = "bfloat16"
    head: str = "float32"
    bf16_resident: bool = True
    gpu_weights: str | None = None
    cpu_weights: str | None = None
    reduced_gpu: str | None = None
    reduced_cpu: str | None = None
    approximate_kernels: bool = False


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

    @property
    def requires(self) -> Mapping[str, tuple[str, ...]]:
        """The device capabilities the architecture needs, per accelerator (``ModelSpec.requires``).

        Gated DeltaNet layers (Qwen3.5) solve triangular systems; on a CPU that needs LAPACK.
        """
        return {"cpu": ("lapack",)} if self.model_type == "qwen3_5_text" else {}


@dataclass(frozen=True)
class ModelSpec:
    """Everything an engine needs to build and run a backbone.

    ``graphs`` names ONNX graphs a package ships (``{"default": path}``), for
    engines that run a graph instead of building the backbone; ``encoder``
    marks a bidirectional encoder whose readout needs hidden states.
    ``kernel_variants`` maps a kernel slot to the variant the model's released
    runtime ran (``KernelSet.use_variants``); a slot without that variant on
    the device runs its reference. ``graph_threads`` caps a named graph's CPU
    intra-op threads where more of them only add wake-ups (a few-millisecond
    forward); other graphs use every configured thread. ``graph_spin_us``
    sets how long a named graph's idle CPU threads spin before they sleep
    (0: never), where the engine's default bound lets them sleep inside a
    run; the engine may shorten it beside other engines' CPU models. ``requires``
    names, per accelerator, the capabilities its devices must report
    (``Accelerator.capabilities``) to serve the model; placement refuses a
    device that lacks one. ``towers`` are a multimodal model's other
    backbones by name (an image or audio encoder next to the text backbone),
    each its own architecture and weights; an encoder batch runs one with
    ``EncoderBatch.tower``.
    """

    name: str
    backbone: BackboneSpec
    dtype: DtypePolicy
    max_input_tokens: int
    graphs: Mapping[str, Path] = field(default_factory=dict)
    encoder: bool = False
    kernel_variants: Mapping[str, str] = field(default_factory=dict)
    graph_threads: Mapping[str, int] = field(default_factory=dict)
    graph_spin_us: Mapping[str, int] = field(default_factory=dict)
    requires: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    towers: Mapping[str, BackboneSpec] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Devices and engines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    """One device an accelerator offers. ``bf16``: the device computes BF16 natively."""

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
    model's ``DtypePolicy`` consents to on its device. ``cpu_neighbors`` names
    the engines of the process's other CPU models (``auto`` where only loading
    can tell): threads an engine leaves spinning after a run slow another
    engine's next forward on those cores.
    """

    graphs: bool = True
    fused_kernels: bool = True
    exact_kernels_only: bool = True
    merge_lora: bool = False
    reduced_precision: bool = False
    threads: int | None = None
    cpu_neighbors: frozenset[str] = frozenset()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoInputs:
    """The videos of a decoder batch: their tower ``features`` (merged, every row's videos, rows in order).

    ``grids`` holds each video's (t, h, w) patch grid in the same order; each
    group of ``h * w`` patches (a pair of frames) takes the next run of the
    batch's ``token_id`` placeholders.
    """

    features: torch.Tensor
    grids: list[tuple[int, int, int]]
    token_id: int


@dataclass
class ImageInputs:
    """The images of a decoder batch, for a model whose ``ModelSpec.towers`` holds a vision tower.

    ``pixel_values`` holds the patch rows of every image of every row, rows in
    order and each row's images in order; ``grids`` each image's (t, h, w)
    patch grid in the same order. The tower's features replace the batch's
    ``token_id`` placeholders, in the same order. ``features``, when given, are
    those features already computed (``EngineModel.tower_features``) and
    ``pixel_values`` is not read; ``videos`` are the rows' video inputs.
    """

    pixel_values: torch.Tensor | None
    grids: list[tuple[int, int, int]]
    token_id: int
    tower: str = "vision"
    features: torch.Tensor | None = None
    videos: VideoInputs | None = None


@dataclass
class ForwardBatch:
    """Padded token rows and the positions whose hidden states the readout needs.

    ``gather`` holds per-row option endpoints padded with 0 to the widest row;
    ``query`` holds one query position per row; ``lengths`` the unpadded lengths.
    ``shared_prefix`` > 0 asks an engine that ``supports_shared_context`` to run the
    rows as one shared-context tree whose first ``shared_prefix`` tokens (common to
    every row, before any endpoint) are computed once. ``images`` are the rows'
    image inputs, for a backbone that reads them.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    gather: torch.Tensor
    query: torch.Tensor
    lengths: list[int]
    shared_prefix: int = 0
    images: ImageInputs | None = None


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
    instead of its own layer stack. ``tower`` runs one of the model's towers
    (``ModelSpec.towers``) on ``graph_inputs`` instead of the backbone, and
    returns its named outputs. ``reduced`` runs the reduced-precision copy
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
    tower: str | None = None


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
    # Whether ``forward`` runs a batch's ``shared_prefix`` once, as a shared-context tree.
    supports_shared_context: ClassVar[bool] = False
    # The spec an engine that builds the backbone loaded; profiles read its config.
    spec: ModelSpec | None = None
    # Set at load when every encoder forward returns each row the same alone or in any batch.
    batch_invariant: bool = False

    @property
    def replays_graphs(self) -> bool:
        """Whether decoder forwards replay captured device graphs."""
        return False

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        """Run a decoder backbone and gather the requested rows (on the device)."""
        raise NotImplementedError(f"{type(self).__name__} has no decoder forward")

    def encode(self, batch: EncoderBatch) -> EncoderOutput:
        """Run an encoder and return hidden states at the requested exits or graph outputs."""
        raise NotImplementedError(f"{type(self).__name__} has no encoder forward")

    def tree(self, batch: TreeBatch) -> TreeOutput:
        """Run a prefix once and every block from it (decoder backbones with a tree forward)."""
        raise NotImplementedError(f"{type(self).__name__} has no tree forward")

    def tower_features(
        self, name: str, pixel_values: torch.Tensor, grids: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        """A tower's merged features for its inputs (``ImageInputs.features``), on the device."""
        raise NotImplementedError(f"{type(self).__name__} has no {name} tower")

    @abstractmethod
    def parameter_count(self) -> int: ...

    def memory_bytes(self) -> int:
        return 0

    def max_forward_tokens(self) -> int | None:
        """Most tokens one forward may hold on this device's kernels; None when they set no limit."""
        return None

    def place(self, module: ModuleT) -> ModuleT:
        """A family's own module (a head) on this device, laid out as the backbone is."""
        return module.to(self.device)

    def autocast(self) -> AbstractContextManager[Any]:
        from contextlib import nullcontext

        return nullcontext()

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release device memory."""


class Engine(ABC):
    """Runs backbones that ``supports`` accepts on an accelerator's devices.

    ``auto_priority`` places the engine in ``--engine auto``'s order, lowest
    first, after the engine a built-in model's table prefers for the device
    class; engines without one (the default) follow, by name.
    """

    name: ClassVar[str]
    auto_priority: ClassVar[int | None] = None

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        """Capability descriptor listed in ``/v1/models``: ``auto_priority``, then architectures, outputs, devices.

        An engine's own descriptor extends this one, so every card shows where
        ``auto`` tries the engine.
        """
        return {"auto_priority": cls.auto_priority}

    @abstractmethod
    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None:
        """None when the engine can run ``spec`` on ``device``, else the reason it cannot."""

    def read(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> Callable[[], EngineModel]:
        """The host work of a load (reading the weights), then the device work that finishes it.

        The runtime runs this before it takes the device (``Accelerator.execute``),
        so the other models of the device keep answering while the weights are
        read, and runs the returned callable as device work. By default all of
        ``load`` is device work.
        """
        return partial(self.load, spec, accelerator, device, options)

    @abstractmethod
    def load(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> EngineModel: ...


class Accelerator(ABC):
    """Devices of one kind, named on ``--device`` as ``<name>[:N]``.

    ``auto_priority`` places the accelerator in ``--device auto``'s order,
    lowest first; None (the default) serves only devices named explicitly.
    """

    name: ClassVar[str]
    validated: ClassVar[bool]
    auto_priority: ClassVar[int | None] = None

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        """Capability descriptor listed in ``/v1/models``; per-device facts come from ``capabilities``."""
        return {"validated": cls.validated, "auto_priority": cls.auto_priority}

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

    def lacks(self, device: DeviceInfo, capability: str) -> str:
        """Why ``device`` can't serve a model that requires ``capability`` (``ModelSpec.requires``), and the remedy."""
        return f"{device.label} lacks {capability}"

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

    def device_fault(self, error: BaseException) -> bool:
        """Whether an error from device work left the device unusable until the process restarts.

        Any other error fails only the batch that raised it.
        """
        return False


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadInfo:
    """A fixed head served on ``/v1/classify``.

    ``kind``: ``sequence`` (softmax distribution), ``scores`` (independent
    sigmoid per label) or ``token`` (spans). ``inputs``: ``text``, ``pair``
    and / or ``grounded``. ``thresholds`` is a packaged operating point (one
    threshold per label); ``window`` the default ``(tokens, overlap)`` for
    ``overflow: window``; ``reduction`` how windows combine when the head
    declares one (``max``, ``span_union``). ``operating_point_sha256`` is the
    digest of the verified policy file the head applies, if any.
    """

    name: str
    kind: HeadKind
    labels: tuple[str, ...]
    inputs: tuple[str, ...] = ("text",)
    default_threshold: float | None = None
    thresholds: tuple[float, ...] | None = None
    overflow: Overflow = "reject"
    window: tuple[int, int] | None = None
    reduction: Reduction | None = None
    operating_point_sha256: str | None = None


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
    """What ``/v1/models`` reports about a loaded model.

    ``modalities`` are the inputs a decisions request may carry: ``text``, and
    ``image`` for a model that reads a request's ``images``.
    """

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
    modalities: tuple[str, ...] = ("text",)


@dataclass(frozen=True)
class SurfaceRequest:
    """A validated request to one surface of one model.

    ``body`` is the JSON body; the runtime has checked its top-level fields
    and parsed the generic options (deadline, profile, return_meta). The
    family validates everything else in ``plan_surface``. ``part`` marks one
    state of a decisions request with several (``states``): only the whole
    request is refused when none of its questions is valid.
    """

    surface: str
    body: dict[str, Any]
    deadline: float | None
    profile: str
    return_meta: bool
    received: float
    part: bool = False

    @property
    def options(self) -> dict[str, Any]:
        options = self.body.get("options")
        return options if isinstance(options, dict) else {}


@dataclass
class SurfacePlan(Generic[ItemT]):
    """A rendered request: work items for the scheduler and private assembly state.

    ``run`` receives the items and ``finish_surface`` their results in order.
    """

    surface: str
    items: list[ItemT]
    input_tokens: int
    state: Any = None


class LoadedModel(ABC, Generic[ItemT, ResultT]):
    """A family's model bound to an engine model, generic over its plan items and their results.

    The runtime calls ``plan_surface`` on the request thread, the scheduler
    calls ``run`` on the model's worker with micro-batches of the plan's
    items, and ``finish_surface`` turns the results into the surface's
    response body: a list in item order, or ``DEADLINE`` when the plan
    expired in the queue; with the result cache an item that expired while
    others were cached is ``DEADLINE`` in the list. Results are shared with
    the cache, so ``finish_surface`` must not mutate them. An item that sets
    ``cache_key`` (a content hash of everything its result depends on) may be
    answered from the per-model result cache instead of a forward. Decision
    families subclass ``plugins.decisions.DecisionModel``, which serves
    ``/v1/decisions`` through their ``plan`` and ``answer``. The golden check
    and the item metrics read every surface through ``golden_values``,
    ``golden_compare`` and ``outcomes``.

    ``fuse_bundled_jobs`` lets the ``exact`` profile run the jobs of one
    bundle as one batch, so a family can compute each distinct input once for
    several heads; decision families keep it off because their released
    numerics batch one request at a time. ``batch_invariant``, set at load,
    says every row's result on this device is the same alone or inside any
    batch (a test must show it); the ``exact`` profile then runs the jobs of
    concurrent requests that are queued together in shared batches, without
    waiting for more. ``packs_rows`` says ``run`` lays rows back to back with
    no padding, so rows of any lengths share those batches at no extra cost.

    ``device_thread`` False says ``run`` never starts a parallel torch op (an
    ONNX Runtime engine with NumPy readouts): the scheduler then calls it on
    the model's worker, or on the request's planning thread while the model is
    idle, instead of handing each batch to the CPU device thread, which saves a
    thread wake-up each way. On any other device each batch still runs through
    the accelerator's ``execute`` (a GPU's device lock).
    """

    info: ModelInfo
    engine_model: EngineModel
    fuse_bundled_jobs: ClassVar[bool] = False
    device_thread: ClassVar[bool] = True
    batch_invariant: bool = False
    packs_rows: bool = False

    @abstractmethod
    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan[ItemT]:
        """Validate and render a request; per-item failures stay in the plan."""

    @abstractmethod
    def run(self, items: list[ItemT]) -> list[ResultT]:
        """One forward over ``items`` in order; per item its readout (decisions: option logits)."""

    def run_shared(self, items: list[ItemT], shared_prefix: int) -> list[ResultT]:
        """``run`` for a shared-context batch: the items' first ``shared_prefix`` tokens run once.

        Only a family whose ``shared_context`` returns positive prefixes runs
        such batches (decoders, as one tree: see ``ForwardBatch``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no shared-context forward"
        )

    def run_approximate(self, items: list[ItemT]) -> list[ResultT]:
        """``run`` for a batch of an approximate profile, where a family may trade exactness for speed."""
        return self.run(items)

    @abstractmethod
    def finish_surface(
        self, plan: SurfacePlan[ItemT], results: Results[ResultT]
    ) -> dict[str, Any]:
        """The response body without ``model``, ``usage`` and ``meta`` (the runtime adds those)."""

    def golden_values(self, surface: str, response: dict[str, Any]) -> dict[str, Any]:
        """A golden response's comparable values by stable key; by default its numbers (``readiness.flatten``).

        The golden check requires two runs to give equal values, and these are
        what ``tools/golden_answers.py`` records as the reference.
        """
        from ..supervision.readiness import flatten

        return flatten(surface, response)

    def golden_compare(
        self,
        surface: str,
        values: dict[str, Any],
        reference: dict[str, Any],
        tolerance: float,
    ) -> tuple[int, int] | None:
        """(checked, matched) of golden values against the device class's reference, or None if they are malformed.

        ``reference`` is empty when none is recorded, which checks nothing.
        By default every value must be a finite number, and each reference
        value is matched within ``tolerance`` by the value of its key.
        """
        from ..supervision.readiness import compare_numbers

        return compare_numbers(values, reference, tolerance)

    def outcomes(self, surface: str, body: dict[str, Any]) -> Iterable[tuple[str, str]]:
        """(type, outcome) of each item of a finished response body, for the runtime's item metrics.

        The outcome is ``answered`` or the item's error code; a surface that
        reports no items (the default) counts nothing.
        """
        return ()

    def forward_token_budget(self) -> int | None:
        """Most padded tokens one forward may hold; None when nothing limits it."""
        return None

    def shared_context(
        self, items: list[ItemT], token_budget: int | None
    ) -> int | None:
        """How a job's items share context on the shared-context path.

        A positive value runs them through ``run_shared(items, value)``,
        0 runs them exactly, and None lets the profile find the common token
        prefix of decision items itself.
        """
        return None

    def exact_batches(self, items: list[ItemT]) -> list[list[int]] | None:
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
    """A model format's task contract: recognise, verify, describe and load its packages.

    ``builtin_table`` names the module whose ``MODELS`` lists the family's
    pinned first-party models (``registry.tables.common.BuiltinModel``), which
    the runtime resolves by name and gates on their recorded golden answers
    (``registry.builtin``). ``fixture_writer`` names the module that writes the
    family's tiny random-weight packages for tests and the CPU E2E profile:
    ``write_fixture(output, variant, seed)`` and ``VARIANTS``, the first the
    default. Both are imported only when first needed.
    """

    name: ClassVar[str]
    surfaces: ClassVar[frozenset[str]]
    builtin_table: ClassVar[str | None] = None
    fixture_writer: ClassVar[str | None] = None

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
    ) -> LoadedModel[Any, Any]: ...

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        """Golden requests (``{surface, body, expected}``, ``expected`` by device class) that gate readiness."""
        return []

    def kernel_choices(
        self, package: VerifiedPackage, device: DeviceInfo
    ) -> dict[str, Any]:
        """Autotuned-kernel configurations this family recorded for the model on the device's class.

        The runtime pins a built-in model's recorded choices itself; a family
        overrides this only for packages that carry their own.
        """
        return {}


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

    items: list[WorkItem]
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

    def items(self) -> list[WorkItem]:
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

    @classmethod
    def from_config(cls, config: ServeConfig) -> Profile:
        """The profile a process's options ask for; one that reads none ignores them."""
        return cls()

    def engine_options(self, base: EngineOptions) -> EngineOptions:
        return base

    def available(self, model: LoadedModel[Any, Any]) -> str | None:
        """None when the profile can run on ``model``, else the reason it cannot."""
        return None

    def bind(self, model: LoadedModel[Any, Any]) -> None:  # noqa: B027 - optional hook
        """Take what planning needs from the model this instance serves, once ``available`` passed."""

    @abstractmethod
    def plan(self, jobs: list[Job], token_budget: int | None) -> list[Batch]:
        """Turn pending jobs into forwards; every item of every job must appear once."""
