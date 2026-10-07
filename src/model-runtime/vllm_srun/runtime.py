"""The runtime process: load its models through the plugin layers, then answer requests.

A process serves one or more models (``ServeConfig.served_models``). Each
model is resolved, verified, placed and loaded in turn by a background
thread, so ``/health`` answers immediately; a model is served only after it
is verified, loaded and has passed its golden check, and a model that fails
to load does not stop the others. Every model has its own scheduler and
worker. Requests name their model (optional while a process serves one);
``/v1/bundle`` fans tasks out to their models' schedulers at once.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any

from .accel.autotune import KernelChoices, freeze_autotune
from .config import ONNX_RUNTIME_SPIN_COUNT, ModelConfig, ServeConfig
from .errors import (
    PackageError,
    RuntimeServiceError,
    UnsupportedDeviceError,
    VerificationError,
)
from .placement import Placement, check_device, device_kind, place
from .plugins import registry
from .plugins.base import (
    DEADLINE,
    SURFACES,
    DeviceInfo,
    EngineOptions,
    Expired,
    LoadedModel,
    ModelFamily,
    ModelSpec,
    PackageRef,
    Profile,
    RegistryOptions,
    Results,
    SurfacePlan,
    SurfaceRequest,
    UnsupportedSurfaceError,
    VerifiedPackage,
)
from .registry import builtin
from .registry.resolve import resolve
from .scheduler.scheduler import Scheduler, SchedulerLimits
from .supervision.metrics import RuntimeMetrics
from .supervision.readiness import STATES, Health, golden_check
from .timing import RunTiming, ServerTiming

log = logging.getLogger("vllm_srun")
AUTO_ENGINE = "auto"


def _auto_rank(engine: str, preferred: str | None) -> tuple[int, int, str]:
    """Where ``auto`` tries an engine: ``preferred``, then by ``Engine.auto_priority``, then by name.

    An engine whose plugin fails to load ranks last, so only reaching it fails.
    """
    if engine == preferred:
        return 0, 0, engine
    try:
        priority = registry.plugin("engines", engine).load().auto_priority
    except Exception:
        priority = None
    return (1, priority, engine) if priority is not None else (2, 0, engine)


def choose_engine(
    name: str, spec: ModelSpec, device: DeviceInfo, preferred: str | None = None
) -> tuple[str, Any]:
    """The named engine, or for ``auto`` the first that runs the spec on the device.

    ``auto`` tries ``preferred`` (the built-in table's engine for the device
    class) first, then the engines in ``_auto_rank`` order.
    """
    candidates = [name]
    if name == AUTO_ENGINE:
        candidates = sorted(
            registry.names("engines"), key=lambda n: _auto_rank(n, preferred)
        )
    reasons = []
    for candidate in candidates:
        engine = registry.instantiate("engines", candidate)
        reason = engine.supports(spec, device)
        if reason is None:
            return candidate, engine
        reasons.append(f"{candidate}: {reason}")
    raise RuntimeError(f"no engine can run {spec.name}: {'; '.join(reasons)}")


def planned_engine(config: ModelConfig) -> str:
    """The engine a served model will run on the CPU, known before anything loads; ``auto`` if only loading can tell.

    A named engine stands. A built-in model's ``auto`` is what
    ``choose_engine`` will pick: the table's CPU engine, else its family's only
    engine (the descriptor's ``engines``).
    """
    if config.engine != AUTO_ENGINE:
        return config.engine
    known = builtin.lookup(config.model)
    if known is None:
        return AUTO_ENGINE
    if "cpu" in known.engines:
        return known.engines["cpu"]
    try:
        family = registry.plugin("families", known.family).load()
    except (KeyError, ImportError):
        return AUTO_ENGINE
    engines = family.descriptor().get("engines", [])
    return engines[0] if len(engines) == 1 else AUTO_ENGINE


def freeze_heap() -> None:
    """Collect once, then leave every object that exists now out of later collections.

    Loading leaves the frameworks', tokenizers' and models' long-lived objects
    behind. A full collection walks all of them (83-131 ms in a process that
    serves Omni Nano and Mini) and stalls whichever request triggers it; once
    they are frozen it walks only what requests allocated since.
    """
    gc.collect()
    gc.freeze()


__all__ = [
    "DEADLINE",
    "Runtime",
    "ServedModel",
    "freeze_heap",
    "planned_engine",
    "with_overrides",
]

GENERIC_OPTIONS = {"deadline_ms", "profile", "return_meta"}
SURFACE_FIELDS = {
    "decisions": {"model", "state", "questions", "options"},
    "classify": {"model", "input", "head", "options"},
    "embeddings": {
        "model",
        "input",
        "dimensions",
        "encoding_format",
        "layer",
        "input_type",
        "user",
        "options",
    },
    "rerank": {
        "model",
        "query",
        "documents",
        "top_n",
        "return_documents",
        "layer",
        "dimensions",
        "options",
    },
}
SURFACE_OPTIONS = {
    "decisions": {"max_tokens"},
    "classify": {"overflow", "max_tokens", "window", "threshold", "return_tokens"},
    "embeddings": {"overflow", "max_tokens"},
    "rerank": {"overflow", "max_tokens"},
}
STATE_ORDER = {state: index for index, state in enumerate(STATES)}
# Requests up to this many encoded bytes are planned on the event loop: their
# rendering costs less than a hop to a worker thread.
INLINE_PLAN_BYTES = 16 << 10


@dataclass
class Prepared:
    """A request planned for one model, ready to submit to its scheduler."""

    served: ServedModel
    request: SurfaceRequest
    plan: SurfacePlan[Any]


@dataclass
class _Lookup:
    """A job group's result-cache hits and the items still to run, per member.

    ``futures`` is set when the planning thread already ran the misses
    (``Scheduler.run_now``), ``submitted`` when it tried to. ``run`` records
    the group's forwards.
    """

    values: dict[tuple[int, int], Any]
    misses: list[list[Any]]
    slots: list[list[tuple[int, int, str | None]]]
    submitted: float | None = None
    futures: list[Future[Results[Any]]] | BaseException | None = None
    run: RunTiming = field(default_factory=RunTiming)


class ResultCache:
    """Item results of one model by content key, least recently used out first.

    Keys come from the family (``item.cache_key``) and the profile; a model
    without cacheable items never touches the cache.
    """

    def __init__(self, entries: int):
        self.entries = entries
        self._values: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            if key not in self._values:
                return False, None
            self._values.move_to_end(key)
            return True, self._values[key]

    def put(self, key: str, value: Any) -> None:
        if self.entries <= 0:
            return
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.entries:
                self._values.popitem(last=False)


def _name_of(artifact: str) -> str:
    return Path(artifact.rstrip("/")).name or artifact


class ServedModel:
    """One model of the process: its configuration, loaded model, scheduler and health."""

    def __init__(self, runtime: Runtime, config: ModelConfig):
        self.runtime = runtime
        self.config = config
        self.health = Health()
        self.family: ModelFamily | None = None
        self.package: VerifiedPackage | None = None
        self.model: LoadedModel[Any, Any] | None = None
        self.placement: Placement | None = None
        self.engine = config.engine
        self.profiles: dict[str, Profile] = {}
        self.scheduler: Scheduler | None = None
        self.cache = ResultCache(runtime.config.result_cache_entries)
        self.kernel_choices: KernelChoices | None = None
        self.unpinned: str | None = None

    # -- names ---------------------------------------------------------------

    @property
    def served_id(self) -> str | None:
        if self.config.name:
            return self.config.name
        return self.model.info.id if self.model else None

    @property
    def label(self) -> str:
        return self.served_id or _name_of(self.config.model)

    def names(self) -> set[str]:
        names = {self.config.name, self.config.model, _name_of(self.config.model)}
        if self.model is not None:
            names |= {self.model.info.id, self.model.info.repo}
        return {name for name in names if name}

    def accepts(self, name: str) -> bool:
        if self.model is None and self.config.name is None:
            return name in self.names() or len(self.runtime.served) == 1
        return name in self.names()

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        process = self.runtime.config
        config = self.config
        check_device(config.model, config.device)
        self.health.set("loading", "resolving the model")
        options = RegistryOptions(
            cache_dir=process.cache_dir,
            offline=process.offline,
            base_path=process.base_path,
            accept_licences=process.accept_licences,
            model_options=dict(config.options),
        )
        ref = resolve(
            config.model,
            revision=config.revision,
            cache_dir=process.cache_dir,
            offline=process.offline,
        )
        family = self._family(ref, options)
        ref = family.fetch(ref)
        self.health.set("loading", "verifying the package")
        package = family.verify(ref)
        spec = family.describe(package)
        parameters = package.loaded_parameters or 0
        budget = config.memory_budget_gib or process.memory_budget_gib
        placement = place(spec, config.device, parameters, budget)
        device = placement.device
        self._pin(
            builtin.kernel_choices(
                package.model_sha256, device.accelerator, device.arch
            )
            or family.kernel_choices(package, device)
        )
        known = builtin.lookup(package.ref.repo_id or "")
        preferred = None
        if known is not None and known.revision == package.ref.revision:
            preferred = known.engines.get(placement.device.accelerator)
        self.engine, engine = choose_engine(
            config.engine, spec, placement.device, preferred
        )
        profiles = self._profiles()
        default = profiles[config.profile]
        neighbors = frozenset(
            planned_engine(served)
            for served in process.served_models()
            if served is not config and device_kind(served.device) == "cpu"
        )
        if (
            self.engine == "onnxruntime"
            and placement.device.accelerator == "cpu"
            and neighbors - {self.engine}
            and "GOMP_SPINCOUNT" not in os.environ
        ):
            log.warning(
                "%s runs ONNX Runtime beside other CPU models on libgomp's default spin "
                "count, which slows its runs after their forwards; name engine: onnxruntime "
                "in its configuration or set GOMP_SPINCOUNT=%s",
                self.label,
                ONNX_RUNTIME_SPIN_COUNT,
            )
        engine_options = default.engine_options(
            EngineOptions(threads=process.threads, cpu_neighbors=neighbors)
        )
        self.health.set("loading", f"loading weights on {placement.device.label}")
        pinned = self.kernel_choices

        def execute(work: Any) -> Any:
            if pinned is not None:
                work = partial(pinned.run, work)
            return placement.accelerator.execute(placement.device, work)

        engine_model = execute(
            engine.read(spec, placement.accelerator, placement.device, engine_options)
        )
        try:
            model = execute(lambda: family.load(package, spec, engine_model))
        except BaseException:
            engine_model.close()
            raise
        self.family, self.package, self.placement, self.model = (
            family,
            package,
            placement,
            model,
        )
        try:
            self._start(profiles, process, execute)
        except BaseException:
            self.stop()
            self.scheduler = self.model = None
            raise
        metrics = self.runtime.metrics
        metrics.model_info.labels(
            model=self.label,
            revision=model.info.revision or "local",
            family=family.name,
            engine=self.engine,
            accelerator=placement.accelerator.name,
            device=placement.device.label,
            profile=config.profile,
        ).set(1)
        metrics.model_memory.labels(model=self.label).set(
            model.engine_model.memory_bytes()
        )
        self.health.set("ready", self.unpinned)
        log.info(
            "serving %s on %s (profile %s)",
            self.label,
            placement.device.label,
            config.profile,
        )

    def _start(
        self,
        profiles: dict[str, Profile],
        process: ServeConfig,
        execute: Callable[[Any], Any],
    ) -> None:
        """Bind the profiles, start the worker and pass the golden check."""
        assert self.model is not None and self.placement is not None
        model, placement = self.model, self.placement
        for name, profile in list(profiles.items()):
            unavailable = profile.available(model)
            if unavailable:
                if name == self.config.profile:
                    raise RuntimeError(
                        f"profile {name!r} is unavailable: {unavailable}"
                    )
                profiles.pop(name)
                continue
            profile.bind(model)
        self.profiles = profiles
        self.scheduler = Scheduler(
            model,
            profiles,
            SchedulerLimits(
                max_queue=process.max_queue,
                max_queued_tokens=process.max_queued_tokens,
                batch_window_ms=process.batch_window_ms,
            ),
            observe=self.runtime.metrics.observe,
            execute=(
                execute
                if model.device_thread or placement.device.accelerator != "cpu"
                else self._inline()
            ),
            device_fault=placement.accelerator.device_fault,
        )
        self.scheduler.start()
        self.health.set("warming", "running the golden check")
        assert self.family is not None and self.package is not None
        self.health.golden = golden_check(
            self.golden_surface,
            model.golden_compare,
            self.family.golden(self.package),
            placement.device.accelerator,
        )
        if self.health.golden.status == "failed":
            raise VerificationError(self.health.golden.detail or "golden check failed")
        if self.unpinned and self.health.golden.status == "matched":
            self.health.golden = replace(
                self.health.golden, status="unverified", detail=self.unpinned
            )

    def _inline(self) -> Callable[[Callable[[], Any]], Any] | None:
        """How a model without a device thread runs its batches on the CPU: inline, in its choice scope if pinned."""
        return self.kernel_choices.run if self.kernel_choices is not None else None

    def _pin(self, recorded: dict[str, Any]) -> None:
        """Run the model's device work on its recorded kernel choices, or record why it can't.

        A model whose choices can't be applied still loads, but its golden
        result says ``unverified``: it can't vouch for the released answers.
        """
        self.kernel_choices = self.unpinned = None
        if not recorded:
            return
        choices = KernelChoices(recorded)
        reason = choices.install()
        if reason is None:
            self.kernel_choices = choices
            return
        self.unpinned = f"kernel choices not applied: {reason}"
        log.warning(
            "%s runs without its recorded kernel choices: %s", self.label, reason
        )

    def stop(self) -> None:
        """Stop the worker, then free the model once no forward can still use it."""
        stopped = self.scheduler.stop() if self.scheduler is not None else True
        if self.model is not None:
            if stopped:
                self.model.close()
            else:
                log.warning(
                    "%s: a forward is still running; the model stays open", self.label
                )

    def _family(self, ref: PackageRef, options: RegistryOptions) -> ModelFamily:
        names = (
            [self.config.family] if self.config.family else registry.names("families")
        )
        for name in names:
            family: ModelFamily = registry.plugin("families", name).load()(options)
            if family.detect(ref):
                return family
        raise RuntimeError(
            f"no installed model family recognises {self.config.model!r}"
        )

    def _profiles(self) -> dict[str, Profile]:
        """``exact`` (golden checks run on it) and the model's own profile."""
        names = dict.fromkeys(("exact", self.config.profile))
        return {
            name: registry.plugin("profiles", name)
            .load()
            .from_config(self.runtime.config)
            for name in names
        }

    def _golden(self, surface: str, body: dict[str, Any]) -> dict[str, Any]:
        """A golden request's response body on the exact profile, before readiness."""
        assert self.model is not None
        request = SurfaceRequest(surface, body, None, "exact", False, time.monotonic())
        plan = self.model.plan_surface(surface, request)
        results = self.submit_items(plan.items, None, "exact").result()
        return self.model.finish_surface(plan, results)

    def golden_surface(self, surface: str, body: dict[str, Any]) -> dict[str, Any]:
        """The comparable values of a golden response (``LoadedModel.golden_values``)."""
        assert self.model is not None
        return self.model.golden_values(surface, self._golden(surface, body))

    # -- requests ------------------------------------------------------------

    def require_ready(self) -> None:
        if not self.health.ready or self.model is None:
            raise RuntimeServiceError(
                "not_ready", f"model {self.label} is {self.health.state}"
            )

    def submit_items(
        self, items: list[Any], deadline: float | None, profile: str
    ) -> Future[Results[Any]]:
        return self.submit_items_group([items], [deadline], profile)[0]

    def submit_items_group(
        self,
        item_lists: list[list[Any]],
        deadlines: list[float | None],
        profile: str,
        timing: RunTiming | None = None,
    ) -> list[Future[Results[Any]]]:
        if self.scheduler is None:
            raise RuntimeServiceError("not_ready", f"the model is {self.health.state}")
        return self.scheduler.submit_group(
            item_lists, deadlines=deadlines, profile=profile, timing=timing
        )

    def run_items_now(
        self,
        item_lists: list[list[Any]],
        deadlines: list[float | None],
        profile: str,
        timing: RunTiming | None = None,
    ) -> list[Future[Results[Any]]] | None:
        """Run the group on this thread if the model's scheduler is idle (never on the event loop)."""
        if self.scheduler is None:
            return None
        return self.scheduler.run_now(
            item_lists, deadlines=deadlines, profile=profile, timing=timing
        )

    def meta(
        self, profile_name: str, queue_ms: float, compute_ms: float
    ) -> dict[str, Any]:
        assert self.model is not None and self.placement is not None
        profile = self.profiles.get(profile_name)
        return {
            "revision": self.model.info.revision,
            "model_sha256": self.model.info.model_sha256,
            "profile": profile_name,
            "numerics": profile.numerics if profile else "exact",
            "engine": self.engine,
            "accelerator": self.placement.accelerator.name,
            "device": self.placement.device.label,
            "queue_ms": round(queue_ms, 3),
            "compute_ms": round(compute_ms, 3),
        }

    def device_failure(self) -> BaseException | None:
        return self.scheduler.failure if self.scheduler else None

    # -- description ---------------------------------------------------------

    def card(self, plugins: list[dict[str, Any]]) -> dict[str, Any]:
        if self.model is None or self.placement is None or self.package is None:
            return {
                "id": self.config.name or self.config.model,
                "object": "model",
                "family": self.family.name if self.family else "unknown",
                "surfaces": [],
                "ready": False,
                "status": self.health.state,
                "reason": self.health.reason,
                "golden": self.health.golden.describe(),
                "plugins": plugins,
            }
        info = self.model.info
        card: dict[str, Any] = {
            "id": self.served_id,
            "object": "model",
            "owned_by": "vllm-sr",
            "family": info.family,
            "repo": info.repo,
            "revision": info.revision,
            "model_sha256": info.model_sha256,
            "manifest_sha256": info.manifest_sha256,
            "surfaces": list(info.surfaces),
            "question_types": list(info.question_types),
            "limits": dict(info.limits),
            "licence": info.licence,
            "profile": self.config.profile,
            "profiles": [
                {
                    "name": name,
                    "numerics": profile.numerics,
                    "description": profile.description,
                }
                for name, profile in self.profiles.items()
            ],
            "engine": self.engine,
            "accelerator": self.placement.accelerator.name,
            "accelerator_validated": bool(self.placement.accelerator.validated),
            "device": self.placement.device.label,
            "dtype": info.dtype,
            "parameters": info.parameters,
            "ready": self.health.ready,
            "status": self.health.state,
            "reason": self.health.reason,
            "golden": self.health.golden.describe(),
            "plugins": plugins,
        }
        if info.heads:
            card["heads"] = [
                {
                    "name": head.name,
                    "kind": head.kind,
                    "labels": list(head.labels),
                    "inputs": list(head.inputs),
                    "default_threshold": head.default_threshold,
                    "thresholds": list(head.thresholds) if head.thresholds else None,
                    "overflow": head.overflow,
                    "window": (
                        {"tokens": head.window[0], "overlap": head.window[1]}
                        if head.window
                        else None
                    ),
                    "reduction": head.reduction,
                    "operating_point_sha256": head.operating_point_sha256,
                }
                for head in info.heads
            ]
        if info.embedding is not None:
            card["embedding"] = {
                "dimensions": list(info.embedding.dimensions),
                "layers": list(info.embedding.layers),
                "modalities": list(info.embedding.modalities),
                "normalized": info.embedding.normalized,
                "pooling": info.embedding.pooling,
                "input_types": list(info.embedding.input_types),
            }
        if info.rerank is not None:
            card["rerank"] = {
                "exits": [
                    {"layer": layer, "dimension": dim}
                    for layer, dim in info.rerank.exits
                ],
                "default": {
                    "layer": info.rerank.default[0],
                    "dimension": info.rerank.default[1],
                },
            }
        if info.presets:
            card["presets"] = list(info.presets)
        return card


class ProcessHealth:
    """The process's readiness: ready when every model is ready."""

    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    @property
    def _models(self) -> list[ServedModel]:
        return self._runtime.served

    @property
    def ready(self) -> bool:
        return all(served.health.ready for served in self._models)

    @property
    def state(self) -> str:
        states = [served.health.state for served in self._models]
        if all(state == "ready" for state in states):
            return "ready"
        if any(state in ("starting", "loading", "warming") for state in states):
            return min(
                (s for s in states if s in ("starting", "loading", "warming")),
                key=STATE_ORDER.__getitem__,
            )
        if all(state == "failed" for state in states):
            return "failed"
        return "degraded"

    @property
    def reason(self) -> str | None:
        for served in self._models:
            if not served.health.ready:
                if len(self._models) == 1:
                    return served.health.reason
                return f"{served.label}: {served.health.reason or served.health.state}"
        return None

    def set(self, state: str, reason: str | None = None) -> None:
        for served in self._models:
            served.health.set(state, reason)

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            served.label: {
                "status": served.health.state,
                "reason": served.health.reason,
            }
            for served in self._models
        }


class Runtime:
    def __init__(self, config: ServeConfig, metrics: RuntimeMetrics | None = None):
        self.config = config
        self.metrics = metrics or RuntimeMetrics()
        self.served = [ServedModel(self, model) for model in config.served_models()]
        labels = [served.config.name for served in self.served if served.config.name]
        if len(labels) != len(set(labels)):
            raise ValueError("served model names must be unique")
        self.health = ProcessHealth(self)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self, *, background: bool = True) -> None:
        if background:
            self._thread = threading.Thread(
                target=self._load_all, name="vllm-srun-load", daemon=True
            )
            self._thread.start()
        else:
            self.load()

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.health.ready

    def stop(self) -> None:
        self._stopping.set()
        for served in self.served:
            served.stop()

    def load(self) -> None:
        """Load every model in order, then ``freeze_heap``; the first failure is raised (foreground start)."""
        if self.config.autotune_cache:
            freeze_autotune(self.config.autotune_cache)
        for served in self.served:
            served.load()
            self._ready_gauge()
        freeze_heap()

    def _load_all(self) -> None:
        """Load every model in order, then reload the ones that failed with back-off.

        A model whose package or golden answers are wrong, or that no
        device it may use can serve (``UnsupportedDeviceError``), stays failed
        at once. Any other failure (no device with enough free memory, a
        download, a busy device) is retried ``load_attempts`` times in all,
        waiting ``load_retry_seconds`` and doubling up to 300 s, while the
        model reports ``loading``. The other models of its device keep serving
        while an attempt reads the weights (``Engine.read``) and wait for its
        device work: the copy to the device, the family's load and the golden
        check. Every pass ends with ``freeze_heap``.
        """
        if self.config.autotune_cache:
            freeze_autotune(self.config.autotune_cache)
        pending = list(self.served)
        attempts = max(1, self.config.load_attempts)
        for attempt in range(1, attempts + 1):
            retry = []
            for served in pending:
                if self._stopping.is_set():
                    return
                try:
                    served.load()
                except (PackageError, UnsupportedDeviceError, VerificationError) as exc:
                    log.error("loading %s failed: %s", served.label, exc)
                    served.health.set("failed", f"{type(exc).__name__}: {exc}")
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    if attempt == attempts:
                        log.error("loading %s failed: %s", served.label, failure)
                        served.health.set("failed", failure)
                    else:
                        log.warning(
                            "loading %s failed, retrying: %s", served.label, failure
                        )
                        served.health.set(
                            "loading",
                            f"retrying after {failure} (attempt {attempt} of {attempts})",
                        )
                        retry.append(served)
                self._ready_gauge()
            freeze_heap()
            if not retry:
                return
            delay = min(self.config.load_retry_seconds * 2 ** (attempt - 1), 300.0)
            if self._stopping.wait(delay):
                return
            pending = retry

    def _ready_gauge(self) -> None:
        self.metrics.ready.set(1 if self.health.ready else 0)

    # -- routing -------------------------------------------------------------

    def lookup(self, name: Any) -> ServedModel:
        """The model a request names; optional while the process serves one model."""
        if name is None:
            if len(self.served) == 1:
                return self.served[0]
            raise RuntimeServiceError(
                "invalid_request",
                "model is required: this runtime serves "
                + ", ".join(sorted(served.label for served in self.served)),
            )
        if not isinstance(name, str):
            raise RuntimeServiceError("invalid_request", "model must be a string")
        for served in self.served:
            if served.accepts(name):
                return served
        raise RuntimeServiceError(
            "model_not_found",
            f"this runtime serves {', '.join(sorted(s.label for s in self.served))}, not {name}",
        )

    def _options(
        self,
        served: ServedModel,
        surface: str,
        body: dict[str, Any],
        received: float | None = None,
    ) -> tuple[float | None, str, bool, float]:
        """Generic options; ``received`` is when the HTTP request arrived (default now).

        Every task of one bundle shares ``received``, so equal ``deadline_ms``
        values give equal deadlines.
        """
        options = body.get("options") or {}
        allowed = GENERIC_OPTIONS | SURFACE_OPTIONS[surface]
        if not isinstance(options, dict) or set(options) - allowed:
            raise RuntimeServiceError(
                "invalid_request", f"options accepts only {sorted(allowed)}"
            )
        if received is None:
            received = time.monotonic()
        deadline = None
        if "deadline_ms" in options:
            value = options["deadline_ms"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise RuntimeServiceError(
                    "invalid_request", "options.deadline_ms must be a positive number"
                )
            deadline = received + float(value) / 1000.0
        profile = options.get("profile", served.config.profile)
        if not isinstance(profile, str) or (
            served.profiles and profile not in served.profiles
        ):
            raise RuntimeServiceError(
                "invalid_request",
                f"options.profile must be one of {sorted(served.profiles) or ['exact']}",
            )
        return_meta = options.get("return_meta", False)
        if not isinstance(return_meta, bool):
            raise RuntimeServiceError(
                "invalid_request", "options.return_meta must be a boolean"
            )
        return deadline, profile, return_meta, received

    # -- every surface -------------------------------------------------------

    def prepare(
        self, surface: str, body: Any, received: float | None = None
    ) -> Prepared:
        """Validate a surface request and plan it for its model (runs off the event loop)."""
        if surface not in SURFACES:
            raise RuntimeServiceError("invalid_request", f"unknown surface {surface!r}")
        if len(self.served) == 1:
            self.served[0].require_ready()
        if not isinstance(body, dict):
            raise RuntimeServiceError(
                "invalid_request", "the request body must be a JSON object"
            )
        unknown = set(body) - SURFACE_FIELDS[surface]
        if unknown:
            raise RuntimeServiceError(
                "invalid_request", f"unknown request fields: {sorted(unknown)}"
            )
        served = self.lookup(body.get("model"))
        served.require_ready()
        assert served.model is not None
        if surface not in served.model.info.surfaces:
            raise RuntimeServiceError(
                "unsupported_surface",
                f"model {served.label} does not serve /v1/{surface}",
            )
        deadline, profile, return_meta, received = self._options(
            served, surface, body, received
        )
        request = SurfaceRequest(
            surface, body, deadline, profile, return_meta, received
        )
        try:
            plan = served.model.plan_surface(surface, request)
        except UnsupportedSurfaceError as exc:
            raise RuntimeServiceError("unsupported_surface", str(exc)) from exc
        except ValueError as exc:
            raise RuntimeServiceError("invalid_request", str(exc)) from exc
        return Prepared(served, request, plan)

    def finish(
        self, prepared: Prepared, results: Any, queue_ms: float, compute_ms: float
    ) -> dict[str, Any]:
        served, request, plan = prepared.served, prepared.request, prepared.plan
        assert served.model is not None
        body = served.model.finish_surface(plan, results)
        for kind, outcome in served.model.outcomes(request.surface, body):
            self.metrics.questions.labels(type=kind, outcome=outcome).inc()
        response: dict[str, Any] = {"model": served.served_id, **body}
        response.setdefault(
            "usage", {"input_tokens": plan.input_tokens, "output_tokens": 0}
        )
        if request.return_meta:
            response["meta"] = {
                **served.meta(request.profile, queue_ms, compute_ms),
                **response.get("meta", {}),
            }
        self.metrics.input_tokens.inc(plan.input_tokens)
        return response

    async def call(
        self,
        surface: str,
        body: Any,
        size: int | None = None,
        timing: ServerTiming | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Serve one surface request; returns (HTTP status, body).

        ``size`` is the request's encoded size: small requests are planned on
        the event loop, larger ones on a worker thread. ``timing`` receives
        the request's phases.
        """
        return (await self._serve([(surface, body)], size, timing))[0]

    async def bundle(
        self,
        body: Any,
        size: int | None = None,
        timing: ServerTiming | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Serve every task of a bundle at once; results keep task order."""
        try:
            tasks = self._bundle_tasks(body)
        except RuntimeServiceError as exc:
            return exc.status, exc.body()
        outcomes = await self._serve(
            [(surface, task_body) for _, surface, task_body in tasks], size, timing
        )
        results = []
        for (task_id, surface, _), (status, response) in zip(
            tasks, outcomes, strict=True
        ):
            if status == HTTPStatus.OK:
                results.append({"id": task_id, "status": status, surface: response})
            else:
                results.append({"id": task_id, "status": status, **response})
        self.metrics.bundle_tasks.observe(len(tasks))
        return 200, {"results": results}

    def _prepare_outcome(
        self, surface: str, body: Any, received: float
    ) -> Prepared | tuple[int, dict[str, Any]]:
        try:
            return self.prepare(surface, body, received)
        except Exception as exc:
            return _error_outcome(exc)

    def _prepare_and_run(
        self, surface: str, body: Any, received: float, timing: ServerTiming
    ) -> tuple[Prepared | tuple[int, dict[str, Any]], _Lookup | None]:
        """Plan one request off the event loop and run it here if its model is idle."""
        prepared = self._prepare_outcome(surface, body, received)
        timing.tokenize = time.monotonic() - received
        if not isinstance(prepared, Prepared):
            return prepared, None
        lookup = self._lookup([(0, prepared)])
        lookup.submitted = time.monotonic()
        try:
            lookup.futures = prepared.served.run_items_now(
                lookup.misses,
                [prepared.request.deadline],
                prepared.request.profile,
                lookup.run,
            )
        except Exception as exc:
            lookup.futures = exc
        if lookup.futures is None:
            lookup.submitted = None
        return prepared, lookup

    async def _serve(
        self,
        requests: list[tuple[str, Any]],
        size: int | None = None,
        timing: ServerTiming | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Plan every request, run each model's share as one job group, finish in order.

        A group holds a model's tasks of one profile, whatever their deadlines:
        each job keeps its own. A lone request planned off the event loop runs
        on its planning thread when its model's scheduler is idle.
        """
        if timing is None:
            timing = ServerTiming()
        received = time.monotonic()
        lookups: list[_Lookup | None] = [None] * len(requests)
        if size is not None and size <= INLINE_PLAN_BYTES:
            planned = [
                self._prepare_outcome(surface, body, received)
                for surface, body in requests
            ]
            timing.tokenize = time.monotonic() - received
        elif len(requests) == 1:
            from starlette.concurrency import run_in_threadpool

            prepared, lookups[0] = await run_in_threadpool(
                self._prepare_and_run, *requests[0], received, timing
            )
            planned = [prepared]
        else:
            from starlette.concurrency import run_in_threadpool

            planned = await asyncio.gather(
                *(
                    run_in_threadpool(self._prepare_outcome, surface, body, received)
                    for surface, body in requests
                )
            )
            timing.tokenize = time.monotonic() - received
        if lookups[0] is not None:
            first = planned[0]
            assert isinstance(first, Prepared)
            outcomes: list[tuple[int, dict[str, Any]] | None] = [None]
            await self._run_group([(0, first)], outcomes, timing, lookups[0])
            return [outcome for outcome in outcomes if outcome is not None]
        outcomes = [None] * len(requests)
        groups: dict[tuple[int, str], list[tuple[int, Prepared]]] = {}
        for index, prepared in enumerate(planned):
            if isinstance(prepared, Prepared):
                key = (id(prepared.served), prepared.request.profile)
                groups.setdefault(key, []).append((index, prepared))
            else:
                outcomes[index] = prepared
        await asyncio.gather(
            *(self._run_group(members, outcomes, timing) for members in groups.values())
        )
        return [outcome for outcome in outcomes if outcome is not None]

    def _lookup(self, members: list[tuple[int, Prepared]]) -> _Lookup:
        served = members[0][1].served
        request = members[0][1].request
        values: dict[tuple[int, int], Any] = {}
        misses: list[list[Any]] = []
        slots: list[list[tuple[int, int, str | None]]] = []
        for member, (_, prepared) in enumerate(members):
            items, member_slots = [], []
            for position, item in enumerate(prepared.plan.items):
                key = (
                    getattr(item, "cache_key", None)
                    if served.cache.entries > 0
                    else None
                )
                if key is not None:
                    key = f"{request.profile}:{key}"
                    hit, value = served.cache.get(key)
                    self.metrics.result_cache.labels(
                        model=served.label, outcome="hit" if hit else "miss"
                    ).inc()
                    if hit:
                        values[(member, position)] = value
                        continue
                items.append(item)
                member_slots.append((member, position, key))
            misses.append(items)
            slots.append(member_slots)
        return _Lookup(values, misses, slots)

    async def _run_group(
        self,
        members: list[tuple[int, Prepared]],
        outcomes: list[tuple[int, dict[str, Any]] | None],
        timing: ServerTiming,
        lookup: _Lookup | None = None,
    ) -> None:
        """Run one model's job group; every member is answered from its own jobs."""
        served = members[0][1].served
        if lookup is None:
            lookup = self._lookup(members)
        values, slots = lookup.values, lookup.slots
        submitted = lookup.submitted or time.monotonic()
        try:
            futures = lookup.futures
            if isinstance(futures, BaseException):
                raise futures
            if futures is None:
                futures = served.submit_items_group(
                    lookup.misses,
                    [prepared.request.deadline for _, prepared in members],
                    members[0][1].request.profile,
                    lookup.run,
                )
        except Exception as exc:
            for index, _ in members:
                outcomes[index] = _error_outcome(exc)
            return
        results = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures),
            return_exceptions=True,
        )
        if any(isinstance(result, BaseException) for result in results):
            failure = served.device_failure()
            if failure is not None:
                self.degrade(f"{type(failure).__name__}: {failure}", served)
        finished = time.monotonic()
        compute_ms = (finished - submitted) * 1000.0
        for member, (index, prepared) in enumerate(members):
            member_results = results[member]
            if isinstance(member_results, BaseException):
                outcomes[index] = _error_outcome(member_results)
                continue
            for offset, (_, position, key) in enumerate(slots[member]):
                value = (
                    DEADLINE
                    if isinstance(member_results, Expired)
                    else member_results[offset]
                )
                values[(member, position)] = value
                if key is not None and value is not DEADLINE and value is not None:
                    served.cache.put(key, value)
            count = len(prepared.plan.items)
            item_values = [values[(member, position)] for position in range(count)]
            if count and all(value is DEADLINE for value in item_values):
                results_for_member: Any = DEADLINE
            else:
                results_for_member = item_values
            queue_ms = (submitted - prepared.request.received) * 1000.0
            try:
                outcomes[index] = (
                    200,
                    self.finish(prepared, results_for_member, queue_ms, compute_ms),
                )
            except Exception as exc:
                outcomes[index] = _error_outcome(exc)
        timing.ran(lookup.run, submitted, finished, time.monotonic())

    def _bundle_tasks(self, body: Any) -> list[tuple[str, str, dict[str, Any]]]:
        if not isinstance(body, dict) or set(body) - {"tasks", "options"}:
            raise RuntimeServiceError(
                "invalid_request", "a bundle is an object with tasks and options"
            )
        tasks = body.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise RuntimeServiceError(
                "invalid_request", "tasks must be a nonempty list"
            )
        if len(tasks) > self.config.max_bundle_tasks:
            raise RuntimeServiceError(
                "request_too_large",
                f"a bundle holds at most {self.config.max_bundle_tasks} tasks",
            )
        options = body.get("options") or {}
        if not isinstance(options, dict) or set(options) - {"deadline_ms"}:
            raise RuntimeServiceError(
                "invalid_request", "bundle options accept only deadline_ms"
            )
        bundle_deadline = options.get("deadline_ms")
        if bundle_deadline is not None and (
            isinstance(bundle_deadline, bool)
            or not isinstance(bundle_deadline, (int, float))
            or not math.isfinite(bundle_deadline)
            or bundle_deadline <= 0
        ):
            raise RuntimeServiceError(
                "invalid_request", "options.deadline_ms must be a positive number"
            )
        parsed: list[tuple[str, str, dict[str, Any]]] = []
        seen: set[str] = set()
        for index, task in enumerate(tasks):
            surfaces = (
                [key for key in task if key in SURFACES]
                if isinstance(task, dict)
                else []
            )
            task_id = task.get("id") if isinstance(task, dict) else None
            if (
                not isinstance(task_id, str)
                or not task_id
                or len(surfaces) != 1
                or set(task) - {"id", *SURFACES}
                or not isinstance(task[surfaces[0]], dict)
            ):
                raise RuntimeServiceError(
                    "invalid_request",
                    f"tasks[{index}] needs an id and exactly one surface body "
                    f"({', '.join(SURFACES)})",
                )
            if task_id in seen:
                raise RuntimeServiceError(
                    "invalid_request", f"duplicate task id {task_id!r}"
                )
            seen.add(task_id)
            surface = surfaces[0]
            task_body = dict(task[surface])
            if bundle_deadline is not None:
                task_options = task_body.get("options")
                task_options = (
                    dict(task_options) if isinstance(task_options, dict) else {}
                )
                own = task_options.get("deadline_ms")
                if (
                    not isinstance(own, (int, float))
                    or isinstance(own, bool)
                    or own > bundle_deadline
                ):
                    task_options["deadline_ms"] = bundle_deadline
                task_body["options"] = task_options
            parsed.append((task_id, surface, task_body))
        return parsed

    def device_failure(self) -> BaseException | None:
        for served in self.served:
            failure = served.device_failure()
            if failure is not None:
                return failure
        return None

    def degrade(self, reason: str, served: ServedModel) -> None:
        served.health.set("degraded", reason)
        self._ready_gauge()
        if self.config.exit_on_device_error:
            log.error("device failure, exiting for a clean restart: %s", reason)
            threading.Timer(0.5, lambda: os._exit(3)).start()

    # -- description ---------------------------------------------------------

    def model_cards(self) -> list[dict[str, Any]]:
        plugins = [
            entry.describe()
            for kind in registry.GROUPS
            for entry in registry.discover()[kind].values()
        ]
        return [served.card(plugins) for served in self.served]


def with_overrides(config: ServeConfig, **changes: Any) -> ServeConfig:
    return replace(config, **changes)


def _error_outcome(error: BaseException) -> tuple[int, dict[str, Any]]:
    """The contract's error answer for an exception from planning or running a request."""
    if isinstance(error, RuntimeServiceError):
        return error.status, error.body()
    internal = RuntimeServiceError("internal_error", f"{type(error).__name__}: {error}")
    return 500, internal.body()
