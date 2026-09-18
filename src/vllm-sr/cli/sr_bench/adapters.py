"""Server-owned benchmark plugins; manifests can select IDs, never Python code.

Operator-installed packages expose entry points in ``vllm_sr.sr_bench.adapters``.
Each entry point resolves to a BenchmarkAdapter or a zero-argument factory.
All paid calls must use the supplied context.call so journal and limits apply.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable, Protocol


class AdapterContext(Protocol):
    manifest: dict
    config: dict
    limits: dict

    def call(self, messages, role="subject", target=None, extra_body=None) -> dict: ...
    def cancelled(self) -> bool: ...


@dataclass(frozen=True)
class BenchmarkAdapter:
    id: str
    title: str
    kind: str
    source_url: str
    version: str
    execute: Callable[[dict, AdapterContext], dict]
    preflight: Callable[[dict, dict, dict], None] | None = None
    # Same keyword contract as sources.prepare_dataset; returns its manifest.
    prepare: Callable[..., dict] | None = None
    requires_answer: bool = False
    requires_messages: bool = True
    weight: float = 1.0

    def catalog_entry(self):
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "source_url": self.source_url,
            "adapter_version": self.version,
        }


_adapters: dict[str, BenchmarkAdapter] = {}
_lock = threading.RLock()
_loaded = False


def _validate(adapter):
    if not isinstance(adapter, BenchmarkAdapter) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,63}", adapter.id
    ):
        raise ValueError("Benchmark plugins must provide a valid BenchmarkAdapter")
    if (
        not adapter.version
        or not callable(adapter.execute)
        or not math.isfinite(adapter.weight)
        or adapter.weight <= 0
    ):
        raise ValueError(
            "Benchmark plugin requires version, executor and positive diagnostic weight"
        )
    for callback in (adapter.preflight, adapter.prepare):
        if callback is not None and not callable(callback):
            raise ValueError("Invalid benchmark plugin callback")


def _basic_execute(case, context):
    from .engine import basic_grade

    response = context.call(case["messages"])
    return basic_grade(case, response["final"])


def _external_execute(case, context):
    from .external import execute_case

    return execute_case(case, context)


def _external_preflight(case, manifest, cache):
    from .external import preflight_case

    preflight_case(case, manifest, cache)


def _initialize():
    global _loaded
    with _lock:
        if _loaded:
            return
        from .contracts import BENCHMARKS, BENCHMARK_WEIGHTS

        basic = {"mmlu-pro", "gpqa-diamond", "arc-agi-2"}
        found = {}
        for identity, title, kind, url in BENCHMARKS:
            found[identity] = BenchmarkAdapter(
                identity,
                title,
                kind,
                url,
                "sr-bench-1.0",
                _basic_execute if identity in basic else _external_execute,
                None if identity in basic else _external_preflight,
                requires_answer=identity in basic | {"hle", "simpleqa-verified"},
                requires_messages=identity
                not in {"scicode", "tau3", "terminal-bench-2.1"},
                weight=BENCHMARK_WEIGHTS[identity],
            )
        # Only the operator's installed Python environment supplies entry points.
        for entry in entry_points(group="vllm_sr.sr_bench.adapters"):
            value = entry.load()
            adapter = (
                value()
                if callable(value) and not isinstance(value, BenchmarkAdapter)
                else value
            )
            _validate(adapter)
            if adapter.id in found:
                raise ValueError(
                    "Benchmark plugin ID conflicts with an existing adapter"
                )
            found[adapter.id] = adapter
        _adapters.update(found)
        _loaded = True


def register_adapter(adapter):
    """Register trusted in-process extensions during service initialization."""
    _initialize()
    _validate(adapter)
    with _lock:
        if adapter.id in _adapters:
            raise ValueError("Benchmark adapter already registered")
        _adapters[adapter.id] = adapter


def get_adapter(identity):
    _initialize()
    try:
        return _adapters[identity]
    except KeyError as exc:
        raise ValueError("Unknown registered benchmark adapter") from exc


def list_adapters():
    _initialize()
    return list(_adapters.values())
