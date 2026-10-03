"""Every engine-mode option in one typed object.

``ServeConfig`` holds the process options. A process serves one or more
models, each described by a ``ModelConfig``; the single-model fields of
``ServeConfig`` (``model``, ``revision``, ``device`` ...) describe the only
model when ``models`` is empty, which is the Phase 1 command line.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8100

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
    model: str | None = None
    revision: str | None = None
    device: str = "auto"
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    uds: str | None = None
    profile: str = "exact"
    engine: str = "auto"
    family: str | None = None
    served_model_name: str | None = None
    models: tuple[ModelConfig, ...] = ()
    threads: int | None = None
    memory_budget_gib: float | None = None
    max_queue: int = 256
    max_queued_tokens: int = 1 << 22
    batch_window_ms: float = 2.0
    max_batch_tokens: int = 65_536
    max_request_bytes: int = 8 << 20
    max_bundle_tasks: int = 64
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
        if self.models:
            return self.models
        if not self.model:
            raise ValueError("a runtime needs at least one model")
        return (
            ModelConfig(
                model=self.model,
                revision=self.revision,
                name=self.served_model_name,
                device=self.device,
                profile=self.profile,
                engine=self.engine,
                family=self.family,
                memory_budget_gib=self.memory_budget_gib,
            ),
        )


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
