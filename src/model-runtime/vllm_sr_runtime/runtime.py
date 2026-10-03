"""The runtime: load one model through the plugin layers, then answer requests.

Loading runs in a background thread so ``/health`` answers immediately;
the model is served only after it is verified, loaded and has passed its
golden check.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Any

from .accel.autotune import freeze_autotune
from .config import ServeConfig
from .errors import DEADLINE_EXCEEDED, RuntimeServiceError
from .placement import Placement, place
from .plugins import registry
from .plugins.base import (
    EngineOptions,
    LoadedModel,
    ModelFamily,
    Profile,
    RegistryOptions,
    RequestPlan,
    VerifiedPackage,
)
from .registry.resolve import resolve
from .scheduler.scheduler import DEADLINE, Scheduler, SchedulerLimits
from .supervision.metrics import RuntimeMetrics
from .supervision.readiness import Health, golden_check

log = logging.getLogger("vllm_sr_runtime")

REQUEST_FIELDS = {"model", "state", "questions", "options"}
OPTION_FIELDS = {"deadline_ms", "profile", "return_meta"}


@dataclass
class ParsedRequest:
    state: Any
    questions: dict[str, Any]
    deadline: float | None
    profile: str
    return_meta: bool
    received: float


class Runtime:
    def __init__(self, config: ServeConfig, metrics: RuntimeMetrics | None = None):
        self.config = config
        self.metrics = metrics or RuntimeMetrics()
        self.health = Health()
        self.package: VerifiedPackage | None = None
        self.model: LoadedModel | None = None
        self.placement: Placement | None = None
        self.profiles: dict[str, Profile] = {}
        self.scheduler: Scheduler | None = None
        self.family: ModelFamily | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self, *, background: bool = True) -> None:
        if background:
            self._thread = threading.Thread(
                target=self._load_guarded, name="vllm-sr-runtime-load", daemon=True
            )
            self._thread.start()
        else:
            self.load()

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.health.ready

    def stop(self) -> None:
        if self.scheduler is not None:
            self.scheduler.stop()
        if self.model is not None:
            self.model.close()

    def _load_guarded(self) -> None:
        try:
            self.load()
        except Exception as exc:
            log.error("model load failed: %s", exc)
            self.health.set("failed", f"{type(exc).__name__}: {exc}")

    def load(self) -> None:
        config = self.config
        if config.autotune_cache:
            freeze_autotune(config.autotune_cache)
        self.health.set("loading", "resolving the model")
        options = RegistryOptions(
            cache_dir=config.cache_dir,
            offline=config.offline,
            base_path=config.base_path,
            accept_licences=config.accept_licences,
        )
        ref = resolve(
            config.model,
            revision=config.revision,
            cache_dir=config.cache_dir,
            offline=config.offline,
        )
        family = self._family(ref, options)
        self.health.set("loading", "verifying the package")
        package = family.verify(ref)
        spec = family.describe(package)
        parameters = package.loaded_parameters or 0
        placement = place(spec, config.device, parameters, config.memory_budget_gib)
        engine = registry.instantiate("engines", config.engine)
        reason = engine.supports(spec, placement.device)
        if reason:
            raise RuntimeError(
                f"engine {config.engine!r} cannot run {spec.name}: {reason}"
            )
        profiles = self._profiles()
        default = profiles[config.profile]
        options_engine = default.engine_options(EngineOptions(threads=config.threads))
        self.health.set("loading", f"loading weights on {placement.device.label}")
        engine_model = engine.load(
            spec, placement.accelerator, placement.device, options_engine
        )
        model = family.load(package, spec, engine_model)
        for name, profile in list(profiles.items()):
            unavailable = profile.available(model)
            if unavailable:
                if name == config.profile:
                    raise RuntimeError(
                        f"profile {name!r} is unavailable: {unavailable}"
                    )
                profiles.pop(name)
        self.family, self.package, self.placement, self.model, self.profiles = (
            family,
            package,
            placement,
            model,
            profiles,
        )
        self.scheduler = Scheduler(
            model,
            profiles,
            SchedulerLimits(
                max_queue=config.max_queue,
                max_queued_tokens=config.max_queued_tokens,
                batch_window_ms=config.batch_window_ms,
            ),
            observe=self.metrics.observe,
        )
        self.scheduler.start()
        self.health.set("warming", "running the golden check")
        self.health.golden = golden_check(
            self._golden_run, family.golden(package), placement.device.accelerator
        )
        if self.health.golden.status == "failed":
            raise RuntimeError(self.health.golden.detail or "golden check failed")
        self.metrics.model_info.labels(
            model=model.info.id,
            revision=model.info.revision or "local",
            family=family.name,
            engine=config.engine,
            accelerator=placement.accelerator.name,
            device=placement.device.label,
            profile=config.profile,
        ).set(1)
        self.metrics.ready.set(1)
        self.health.set("ready")
        log.info(
            "serving %s on %s (profile %s)",
            model.info.id,
            placement.device.label,
            config.profile,
        )

    def _family(self, ref, options: RegistryOptions) -> ModelFamily:
        names = (
            [self.config.family] if self.config.family else registry.names("families")
        )
        for name in names:
            family = registry.plugin("families", name).load()(options)
            if family.detect(ref):
                return family
        raise RuntimeError(
            f"no installed model family recognises {self.config.model!r}"
        )

    def _profiles(self) -> dict[str, Profile]:
        exact = registry.instantiate("profiles", "exact")
        profiles: dict[str, Profile] = {"exact": exact}
        if self.config.profile != "exact":
            entry = registry.plugin("profiles", self.config.profile).load()
            if self.config.profile in ("batching", "max_speed"):
                profiles[self.config.profile] = entry(
                    max_batch_tokens=self.config.max_batch_tokens
                )
            else:
                profiles[self.config.profile] = entry()
        return profiles

    def _golden_run(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        parsed = ParsedRequest(state, questions, None, "exact", False, time.monotonic())
        plan = self.plan(parsed)
        results = self.submit(plan, parsed).result()
        return self.assemble(parsed, plan, results, 0.0, 0.0)["answers"]

    # -- requests ------------------------------------------------------------

    @property
    def served_id(self) -> str | None:
        if self.config.served_model_name:
            return self.config.served_model_name
        return self.model.info.id if self.model else None

    def accepts_model(self, name: str) -> bool:
        if self.model is None:
            return False
        names = {self.served_id, self.model.info.id, self.model.info.repo}
        return name in {value for value in names if value}

    def parse(self, body: Any) -> ParsedRequest:
        if not isinstance(body, dict):
            raise RuntimeServiceError(
                "invalid_request", "the request body must be a JSON object"
            )
        unknown = set(body) - REQUEST_FIELDS
        if unknown:
            raise RuntimeServiceError(
                "invalid_request", f"unknown request fields: {sorted(unknown)}"
            )
        if "state" not in body:
            raise RuntimeServiceError("invalid_request", "state is required")
        questions = body.get("questions")
        if (
            not isinstance(questions, dict)
            or not questions
            or any(not isinstance(k, str) or not k for k in questions)
        ):
            raise RuntimeServiceError(
                "invalid_request",
                "questions must be a nonempty mapping of question IDs",
            )
        model = body.get("model")
        if model is not None:
            if not isinstance(model, str):
                raise RuntimeServiceError("invalid_request", "model must be a string")
            if self.model is not None and not self.accepts_model(model):
                raise RuntimeServiceError(
                    "model_not_found",
                    f"this runtime serves {self.served_id}, not {model}",
                )
        options = body.get("options") or {}
        if not isinstance(options, dict) or set(options) - OPTION_FIELDS:
            raise RuntimeServiceError(
                "invalid_request", f"options accepts only {sorted(OPTION_FIELDS)}"
            )
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
        profile = options.get("profile", self.config.profile)
        if not isinstance(profile, str) or (
            self.profiles and profile not in self.profiles
        ):
            raise RuntimeServiceError(
                "invalid_request",
                f"options.profile must be one of {sorted(self.profiles) or ['exact']}",
            )
        return_meta = options.get("return_meta", True)
        if not isinstance(return_meta, bool):
            raise RuntimeServiceError(
                "invalid_request", "options.return_meta must be a boolean"
            )
        return ParsedRequest(
            body["state"], questions, deadline, profile, return_meta, received
        )

    def plan(self, parsed: ParsedRequest) -> RequestPlan:
        if self.model is None:
            raise RuntimeServiceError("not_ready", f"the model is {self.health.state}")
        try:
            return self.model.plan(parsed.state, parsed.questions)
        except ValueError as exc:
            raise RuntimeServiceError("invalid_request", str(exc)) from exc

    def submit(self, plan: RequestPlan, parsed: ParsedRequest) -> Future:
        if self.scheduler is None:
            raise RuntimeServiceError("not_ready", f"the model is {self.health.state}")
        return self.scheduler.submit(
            plan.items, deadline=parsed.deadline, profile=parsed.profile
        )

    def assemble(
        self,
        parsed: ParsedRequest,
        plan: RequestPlan,
        results: Any,
        queue_ms: float,
        compute_ms: float,
    ) -> dict[str, Any]:
        assert self.model is not None and self.placement is not None
        answered: dict[str, dict[str, Any]] = {}
        expired = results is DEADLINE
        for index, item in enumerate(plan.items):
            if expired:
                answered[item.question_id] = {
                    "type": item.task_type,
                    "error": DEADLINE_EXCEEDED,
                }
            else:
                answered[item.question_id] = self.model.answer(item, results[index])
        answers = {}
        for question_id in plan.question_ids:
            answer = plan.errors.get(question_id) or answered[question_id]
            answers[question_id] = answer
            outcome = answer.get("error", "answered")
            self.metrics.questions.labels(
                type=str(answer.get("type")), outcome=outcome
            ).inc()
        response: dict[str, Any] = {
            "model": self.served_id,
            "answers": answers,
            "usage": {"input_tokens": plan.input_tokens, "output_tokens": 0},
        }
        if parsed.return_meta:
            profile = self.profiles.get(parsed.profile)
            response["meta"] = {
                "revision": self.model.info.revision,
                "model_sha256": self.model.info.model_sha256,
                "profile": parsed.profile,
                "numerics": profile.numerics if profile else "exact",
                "engine": self.config.engine,
                "accelerator": self.placement.accelerator.name,
                "device": self.placement.device.label,
                "queue_ms": round(queue_ms, 3),
                "compute_ms": round(compute_ms, 3),
            }
        self.metrics.input_tokens.inc(plan.input_tokens)
        return response

    def device_failure(self) -> BaseException | None:
        return self.scheduler.failure if self.scheduler else None

    def degrade(self, reason: str) -> None:
        self.health.set("degraded", reason)
        self.metrics.ready.set(0)
        if self.config.exit_on_device_error:
            log.error("device failure, exiting for a clean restart: %s", reason)
            threading.Timer(0.5, lambda: os._exit(3)).start()

    # -- description ---------------------------------------------------------

    def model_card(self) -> dict[str, Any]:
        plugins = [
            entry.describe()
            for kind in registry.GROUPS
            for entry in registry.discover()[kind].values()
        ]
        if self.model is None or self.placement is None or self.package is None:
            return {
                "id": self.config.served_model_name or self.config.model,
                "object": "model",
                "family": self.family.name if self.family else "unknown",
                "surfaces": [],
                "ready": False,
                "golden": self.health.golden.describe(),
                "plugins": plugins,
            }
        info = self.model.info
        return {
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
            "engine": self.config.engine,
            "accelerator": self.placement.accelerator.name,
            "accelerator_validated": bool(self.placement.accelerator.validated),
            "device": self.placement.device.label,
            "dtype": info.dtype,
            "parameters": info.parameters,
            "ready": self.health.ready,
            "golden": self.health.golden.describe(),
            "plugins": plugins,
        }


def with_overrides(config: ServeConfig, **changes: Any) -> ServeConfig:
    return replace(config, **changes)
