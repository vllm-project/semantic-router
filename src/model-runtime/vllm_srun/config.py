"""Every engine-mode option in one typed object.

``ServeConfig`` holds the process options. A process serves one or more
models, each described by a ``ModelConfig``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8100
# libgomp's spin count for a process with an ONNX Runtime model on the CPU:
# PyTorch's idle OpenMP threads spin that long after a native forward, and
# libgomp's default (300,000) spins long enough to slow an ONNX Runtime run
# that follows on the same cores.
ONNX_RUNTIME_SPIN_COUNT = "10000"

MODEL_FIELDS = {
    "model",
    "revision",
    "name",
    "device",
    "profile",
    "engine",
    "family",
    "memory_budget_gib",
    "options",
}


@dataclass(frozen=True)
class ModelConfig:
    """One served model: its artifact, served name, placement and family options."""

    model: str
    revision: str | None = None
    name: str | None = None
    device: str = "auto"
    profile: str = "exact"
    engine: str = "auto"
    family: str | None = None
    memory_budget_gib: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServeConfig:
    """Process options; ``memory_budget_gib`` is the budget of a model that sets none."""

    models: tuple[ModelConfig, ...] = ()
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    uds: str | None = None
    threads: int | None = None
    memory_budget_gib: float | None = None
    max_queue: int = 256
    max_queued_tokens: int = 1 << 22
    batch_window_ms: float = 2.0
    max_batch_tokens: int = 65_536
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


def load_models_file(path: str | Path) -> tuple[ModelConfig, ...]:
    """Read a ``--models`` file: ``models: [{model, revision, name, device, ...}]``."""
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
        if config.name:
            if config.name in names:
                raise ValueError(f"{path}: duplicate model name {config.name!r}")
            names.add(config.name)
        models.append(config)
    return tuple(models)
