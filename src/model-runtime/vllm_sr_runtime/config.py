"""Every engine-mode option in one typed object."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PORT = 8100


@dataclass(frozen=True)
class ServeConfig:
    model: str
    revision: str | None = None
    device: str = "auto"
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    uds: str | None = None
    profile: str = "exact"
    engine: str = "native"
    family: str | None = None
    served_model_name: str | None = None
    threads: int | None = None
    memory_budget_gib: float | None = None
    max_queue: int = 256
    max_queued_tokens: int = 1 << 22
    batch_window_ms: float = 2.0
    max_batch_tokens: int = 65_536
    max_request_bytes: int = 8 << 20
    cache_dir: str | None = None
    offline: bool = False
    base_path: str | None = None
    accept_licences: tuple[str, ...] = ()
    log_level: str = "info"
    exit_on_device_error: bool = True
