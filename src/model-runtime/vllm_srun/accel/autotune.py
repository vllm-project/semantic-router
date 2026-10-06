"""Persisted and pinned GPU autotuning, so GPU answers repeat across processes.

FLA's gated-delta kernels choose block sizes and warps by timing them in each
process. Two processes can therefore pick different configurations and answer
the Qwen3.5 sizes differently by rounding. Sharing one autotune cache makes
every process reuse the first process's choices; pinning a model's recorded
choices makes every process run the same kernels without timing any.

FLA's autotuners are process-wide objects, and the released models record
different configurations for the same tuning keys. So pinning routes each
autotuner's lookups per thread: a thread inside a model's ``KernelChoices.scope``
gets that model's configurations, and any other thread the autotuner's own
cache. Several pinned models then share one process, each on its own choices.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
import sys
import threading
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

AUTOTUNE_ENV = "VLLM_SRUN_AUTOTUNE_CACHE"
VERSION = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.M)
# The module whose import defines every autotuned kernel of the chunked gated delta rule.
FLA_GATED_DELTA = "fla.ops.gated_delta_rule"

T = TypeVar("T")


class _Scope(threading.local):
    """The kernel choices of the model whose device work the thread runs, if any."""

    choices: KernelChoices | None = None


_SCOPE = _Scope()
_ROUTING = threading.Lock()


def freeze_autotune(directory: str) -> Path:
    """Record autotune choices in ``directory`` on first use and reuse them afterwards.

    Triton reads these settings when a kernel is decorated, so this must run
    before FLA is imported.
    """
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(path)
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    if "fla" in sys.modules:
        log.warning(
            "FLA was imported before the autotune cache was set; its kernels keep per-process tuning"
        )
    return path


class KernelChoices:
    """One model's recorded FLA kernel configurations, for every tuning key and without timing.

    ``choices`` is one device class's entry of ``registry/kernel_choices.json``:
    the FLA version the choices were recorded with and, per kernel, each
    recorded tuning key with its configuration. A recorded key gets its
    configuration. Any other key gets the first entry, in key-hash order, that
    differs from it only in numbers, else the kernel's first entry. That is how
    FLA's ``FLA_CACHE_MODE=full`` resolves the same entries written as its
    config files, so the released runtime's choices stay the ones that run.
    """

    def __init__(self, choices: dict[str, Any]):
        self.fla: str = choices["fla"]
        self.kernels = {
            name: PinnedKernel(entries) for name, entries in choices["kernels"].items()
        }

    def install(self) -> str | None:
        """Route FLA's autotuners to these choices inside ``scope``; None, or why they can't run here."""
        installed = fla_version()
        if installed is None:
            return "FLA is not installed"
        if installed != self.fla:
            return f"they were recorded with FLA {self.fla}, not {installed}"
        try:
            found = route_autotuners(self.kernels)
        except Exception as exc:
            return f"FLA {installed} failed to import ({type(exc).__name__}: {exc})"
        missing = sorted(set(self.kernels) - found)
        if missing:
            return f"FLA {installed} has no autotuned kernel {', '.join(missing)}"
        return None

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Run FLA's autotuned kernels launched on this thread with these configurations."""
        previous, _SCOPE.choices = _SCOPE.choices, self
        try:
            yield
        finally:
            _SCOPE.choices = previous

    def run(self, work: Callable[[], T]) -> T:
        """``work()`` inside ``scope`` (once per device call, so without a generator)."""
        previous, _SCOPE.choices = _SCOPE.choices, self
        try:
            return work()
        finally:
            _SCOPE.choices = previous

    def pin_thread(self) -> None:
        """Keep these choices on the calling thread from now on (a tool answering one model)."""
        _SCOPE.choices = self


class PinnedKernel:
    """One kernel's recorded entries, resolved per tuning key as described in ``KernelChoices``."""

    def __init__(self, entries: list[dict[str, Any]]):
        self.first = entries[0]["config"]
        self.exact = {fla_key_hash(entry["key"]): entry["config"] for entry in entries}
        self.ordered = sorted(entries, key=lambda entry: fla_key_hash(entry["key"]))
        self.configs: dict[Any, Any] = {}

    def resolve(self, key: Any) -> dict[str, Any]:
        """The recorded configuration that runs ``key``."""
        recorded = self.exact.get(fla_key_hash(key))
        if recorded is not None:
            return recorded
        for entry in self.ordered:
            if fuzzy_match(entry["key"], key):
                return entry["config"]
        return self.first

    def config(self, key: Any) -> Any:
        """The ``triton.Config`` that runs ``key``, built once per key."""
        config = self.configs.get(key)
        if config is None:
            config = self.configs[key] = triton_config(self.resolve(key))
        return config


class RoutedCache(MutableMapping):
    """An FLA autotuner's ``cache``: the scoped model's configurations, else the autotuner's own.

    Triton asks ``key in cache`` and then ``cache[key]`` on every launch, and
    FLA reads its config files only for a key missing from the cache, so a
    pinned thread never times a configuration or reads a file.
    """

    def __init__(self, name: str, own: MutableMapping):
        self.name = name
        self.own = own

    def __contains__(self, key: object) -> bool:
        choices = _SCOPE.choices
        if choices is not None and self.name in choices.kernels:
            return True
        return key in self.own

    def __getitem__(self, key: Any) -> Any:
        choices = _SCOPE.choices
        pinned = None if choices is None else choices.kernels.get(self.name)
        return self.own[key] if pinned is None else pinned.config(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.own[key] = value

    def __delitem__(self, key: Any) -> None:
        del self.own[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.own)

    def __len__(self) -> int:
        return len(self.own)


def route_autotuners(kernels: dict[str, Any]) -> set[str]:
    """Give every FLA autotuner named in ``kernels`` a ``RoutedCache``; the names found.

    FLA keys its config files by kernel name, so every autotuner of a name is
    routed, as FLA would apply a file.
    """
    found = set()
    with _ROUTING:
        for name, autotuners in fla_autotuners().items():
            if name not in kernels:
                continue
            found.add(name)
            for autotuner in autotuners:
                if not isinstance(autotuner.cache, RoutedCache):
                    autotuner.cache = RoutedCache(name, autotuner.cache)
    return found


def fla_autotuners() -> dict[str, list[Any]]:
    """FLA's config-file autotuners by kernel name, after importing the gated delta rule's modules.

    These are the autotuners ``FLA_CACHE_MODE`` applies to. Most sit inside a
    ``triton.heuristics`` wrapper, so module attributes are unwrapped.
    """
    importlib.import_module(FLA_GATED_DELTA)
    from fla.ops.utils.cache import CachedAutotuner
    from triton.runtime.jit import KernelInterface

    found: dict[str, list[Any]] = {}
    seen: set[int] = set()
    for module_name, module in list(sys.modules.items()):
        if module is None or not (
            module_name == "fla" or module_name.startswith("fla.")
        ):
            continue
        for attribute in list(vars(module).values()):
            kernel = attribute
            while isinstance(kernel, KernelInterface) and not isinstance(
                kernel, CachedAutotuner
            ):
                kernel = getattr(kernel, "fn", None)
            if isinstance(kernel, CachedAutotuner) and id(kernel) not in seen:
                seen.add(id(kernel))
                found.setdefault(kernel.kernel_name, []).append(kernel)
    return found


def triton_config(config: dict[str, Any]) -> Any:
    """A recorded configuration as the ``triton.Config`` FLA builds from its config files."""
    import triton
    from packaging import version

    extra = {}
    if version.parse(triton.__version__) >= version.parse("3.5.1"):
        extra = {
            "num_ctas": config["num_ctas"],
            "maxnreg": config.get("maxnreg"),
            "pre_hook": None,
            "ir_override": config.get("ir_override"),
        }
    return triton.Config(
        config["kwargs"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        **extra,
    )


def fuzzy_match(recorded: Any, key: Any) -> bool:
    """FLA's fuzzy key match: equal structure, and numbers match any number (booleans don't)."""
    if _numeric(recorded) and _numeric(key):
        return True
    if isinstance(recorded, (list, tuple)) and isinstance(key, (list, tuple)):
        return len(recorded) == len(key) and all(
            fuzzy_match(a, b) for a, b in zip(recorded, key, strict=True)
        )
    if isinstance(recorded, dict) and isinstance(key, dict):
        return recorded.keys() == key.keys() and all(
            fuzzy_match(recorded[name], key[name]) for name in recorded
        )
    return recorded == key


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fla_version() -> str | None:
    """The version of the FLA that ``import fla`` would load, read without importing it."""
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.origin:
        return None
    match = VERSION.search(Path(spec.origin).read_text(encoding="utf-8"))
    return match.group(1) if match else None


def fla_key_hash(key: Any) -> str:
    """FLA's ``AutotuneKey.key_hash`` of a tuning key."""
    serialized = json.dumps(key, separators=(",", ":"), sort_keys=True)
    return hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()
