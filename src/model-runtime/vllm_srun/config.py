"""Every engine-mode option in one typed object.

``ServeConfig`` holds the process options. A process serves one or more
models, each described by a ``ModelConfig``. ``EngineArgs`` are the engine
arguments of one served model: the process's, unless its ``ModelConfig``
sets its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8100
# libgomp's spin count for a process with an ONNX Runtime model on the CPU:
# PyTorch's idle OpenMP threads spin that long after a native forward, and
# libgomp's default (300,000) spins long enough to slow an ONNX Runtime run
# that follows on the same cores.
ONNX_RUNTIME_SPIN_COUNT = "10000"
DTYPES = ("auto", "float32", "bfloat16", "float16")
# The batch limits of a model that leaves them to the runtime on a device
# without a class of its own (the CPU); GPUs take theirs from
# ``scheduler.calibration.DEVICE_CLASSES``.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 65_536
# Every row costs at least this many padded tokens (planner.padded).
MIN_ROW_TOKENS = 8
# At the default token budget no forward can hold more rows: this default
# never binds there.
DEFAULT_MAX_NUM_SEQS = DEFAULT_MAX_NUM_BATCHED_TOKENS // MIN_ROW_TOKENS
DEFAULT_BATCH_WINDOW_MS = 2.0
MAX_BATCH_WINDOW_MS = 1000.0
# A removed per-model field, and what replaced it.
REMOVED_MODEL_FIELDS = {
    "profile": (
        "batching is on by default; set max_num_seqs, max_num_batched_tokens, "
        "batch_window_ms, enable_shared_context or dtype instead"
    ),
}


@dataclass(frozen=True)
class EngineArgs:
    """How one served model batches requests and which precision it computes in.

    ``max_num_seqs`` is the most rows one forward holds, across requests: a
    decision request takes one row per question, the other surfaces one per
    input or window; 1 runs every row alone. ``max_num_batched_tokens`` bounds
    a forward in tokens: the sum of its rows (each rounded up to 8) for a
    model that packs merged rows into one sequence, else its rows times its
    longest row. None leaves either to the runtime, which picks it by the
    device's class and, on a GPU, adjusts it with a short startup calibration
    (``scheduler.calibration``). ``batch_window_ms`` is how long the scheduler
    waits after the first request to collect more (``max_num_seqs`` above 1
    only). ``dtype`` is the precision the backbone computes in; ``auto`` is
    the one the model was published in. ``enable_shared_context`` runs the
    shared state of a multi-question request once; None leaves it to the
    model's family.
    """

    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    batch_window_ms: float = DEFAULT_BATCH_WINDOW_MS
    dtype: str = "auto"
    enable_shared_context: bool | None = None

    def __post_init__(self) -> None:
        if self.max_num_seqs is not None:
            if isinstance(self.max_num_seqs, bool) or not isinstance(
                self.max_num_seqs, int
            ):
                raise ValueError("max_num_seqs must be an integer")
            if self.max_num_seqs < 1:
                raise ValueError("max_num_seqs must be at least 1")
        if self.max_num_batched_tokens is not None:
            if isinstance(self.max_num_batched_tokens, bool) or not isinstance(
                self.max_num_batched_tokens, int
            ):
                raise ValueError("max_num_batched_tokens must be an integer")
            if self.max_num_batched_tokens < MIN_ROW_TOKENS:
                raise ValueError(
                    f"max_num_batched_tokens must be at least {MIN_ROW_TOKENS}"
                )
        if isinstance(self.batch_window_ms, bool) or not isinstance(
            self.batch_window_ms, (int, float)
        ):
            raise ValueError("batch_window_ms must be a number")
        if not 0 <= self.batch_window_ms <= MAX_BATCH_WINDOW_MS:
            raise ValueError(
                f"batch_window_ms must be between 0 and {MAX_BATCH_WINDOW_MS:g}"
            )
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {', '.join(DTYPES)}")
        if self.enable_shared_context is not None and not isinstance(
            self.enable_shared_context, bool
        ):
            raise ValueError("enable_shared_context must be a boolean")

    def describe(self, shared_context: bool) -> dict[str, Any]:
        """The arguments as ``/v1/models`` reports them, with shared context resolved."""
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "batch_window_ms": float(self.batch_window_ms),
            "dtype": self.dtype,
            "enable_shared_context": shared_context,
        }


ENGINE_ARG_FIELDS = tuple(item.name for item in fields(EngineArgs))
MODEL_FIELDS = {
    "model",
    "revision",
    "name",
    "device",
    "engine",
    "family",
    "memory_budget_gib",
    "options",
    *ENGINE_ARG_FIELDS,
}


@dataclass(frozen=True)
class ModelConfig:
    """One served model: its artifact, served name, placement, family options and engine arguments.

    An engine argument left None takes the process's (``ServeConfig.engine_args``).
    """

    model: str
    revision: str | None = None
    name: str | None = None
    device: str = "auto"
    engine: str = "auto"
    family: str | None = None
    memory_budget_gib: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    batch_window_ms: float | None = None
    dtype: str | None = None
    enable_shared_context: bool | None = None

    def engine_args(self, defaults: EngineArgs) -> EngineArgs:
        """This model's engine arguments: its own where it sets them, else ``defaults``."""
        own = {
            name: getattr(self, name)
            for name in ENGINE_ARG_FIELDS
            if getattr(self, name) is not None
        }
        return replace(defaults, **own)


@dataclass(frozen=True)
class ServeConfig:
    """Process options; ``memory_budget_gib`` and ``engine_args`` apply to the models that set none.

    ``media_workers`` is how many worker processes decode and resize request
    images and videos for the process's models, off the threads that plan
    requests next to the device worker; 0 decodes in the planning threads, None
    picks a count from the CPUs the process may use (``media.auto_workers``).
    """

    models: tuple[ModelConfig, ...] = ()
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    uds: str | None = None
    threads: int | None = None
    memory_budget_gib: float | None = None
    max_queue: int = 256
    max_queued_tokens: int = 1 << 22
    engine_args: EngineArgs = field(default_factory=EngineArgs)
    max_request_bytes: int = 8 << 20
    max_bundle_tasks: int = 64
    load_attempts: int = 5
    load_retry_seconds: float = 5.0
    result_cache_entries: int = 16_384
    cache_dir: str | None = None
    offline: bool = False
    base_path: str | None = None
    accept_licences: tuple[str, ...] = ()
    log_level: str = "info"
    autotune_cache: str | None = None
    exit_on_device_error: bool = True
    media_workers: int | None = None

    def __post_init__(self) -> None:
        if self.media_workers is not None and (
            isinstance(self.media_workers, bool)
            or not isinstance(self.media_workers, int)
            or self.media_workers < 0
        ):
            raise ValueError("media_workers must be a non-negative integer")

    def served_models(self) -> tuple[ModelConfig, ...]:
        """The models this process serves, in load order."""
        if not self.models:
            raise ValueError("a runtime needs at least one model")
        return self.models


def spin_count(models: Sequence[ModelConfig]) -> str | None:
    """``GOMP_SPINCOUNT`` for a process that serves ``models``; None keeps libgomp's default.

    libgomp reads it when PyTorch loads, so the process chooses it from its
    configuration alone. 10,000 when a model may run ONNX Runtime on the CPU
    (``engine: onnxruntime`` on a ``cpu`` or ``auto`` device); native CPU
    forwards run fastest on the default. ``engine: auto`` counts as native:
    it tries ``native`` first, which runs every built-in model.
    """
    for model in models:
        kind = model.device.split(":", 1)[0].strip().lower()
        if model.engine == "onnxruntime" and kind in ("cpu", "auto"):
            return ONNX_RUNTIME_SPIN_COUNT
    return None


def split_revision(value: str) -> tuple[str, str | None]:
    """``repo@revision`` -> (repo, revision); a local path never carries a revision."""
    model, at, revision = value.rpartition("@")
    if at and model and revision and not Path(value).exists():
        return model, revision
    return value, None


def load_models_file(
    path: str | Path, defaults: EngineArgs | None = None
) -> tuple[ModelConfig, ...]:
    """Read a ``--models`` file: ``models: [{model, revision, name, device, ...}]``.

    Each entry's engine arguments are checked against ``defaults`` (the
    process's), which fill the ones it leaves out.
    """
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = document.get("models") if isinstance(document, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a nonempty 'models' list")
    models = []
    names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("model"), str):
            raise ValueError(f"{path}: models[{index}] needs a 'model' string")
        for removed, replacement in REMOVED_MODEL_FIELDS.items():
            if removed in entry:
                raise ValueError(
                    f"{path}: models[{index}].{removed} was removed; {replacement}"
                )
        unknown = set(entry) - MODEL_FIELDS
        if unknown:
            raise ValueError(
                f"{path}: models[{index}] has unknown fields {sorted(unknown)}"
            )
        options = entry.get("options") or {}
        if not isinstance(options, Mapping):
            raise ValueError(f"{path}: models[{index}].options must be a mapping")
        config = ModelConfig(
            **{key: value for key, value in entry.items() if key != "options"},
            options=dict(options),
        )
        try:
            config.engine_args(defaults or EngineArgs())
        except ValueError as exc:
            raise ValueError(f"{path}: models[{index}]: {exc}") from exc
        if config.name:
            if config.name in names:
                raise ValueError(f"{path}: duplicate model name {config.name!r}")
            names.add(config.name)
        models.append(config)
    return tuple(models)
