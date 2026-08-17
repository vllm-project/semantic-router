"""Sampling and distribution checks for workload mixture scenarios."""

from __future__ import annotations

import random
from pathlib import Path

from ..core.request import Request
from .mixture import (
    CompositionWindow,
    MixtureScenario,
    MixtureValidationError,
    WorkloadArchetype,
    _load_cdf_source,
    validate_mixture_scenario,
)
from .mixture_validation import MixtureValidationReport, validate_sample_distribution
from .synthetic import CdfWorkload
from .trace import TraceWorkload


class MixtureSampler:
    """Deterministic sampler for fixed and time-varying mixture scenarios."""

    def __init__(self, scenario: MixtureScenario):
        validate_mixture_scenario(scenario)
        self.scenario = scenario
        self._arrival_rng = random.Random(scenario.seed)
        self._choice_rng = random.Random(scenario.seed + 17)
        self._samplers = {
            archetype.id: _ArchetypeSampler(
                archetype=archetype,
                base_dir=scenario.base_dir,
                seed=scenario.seed + 1009 + index,
            )
            for index, archetype in enumerate(scenario.archetypes)
        }

    def generate(self, lam: float, n_requests: int) -> list[tuple[float, Request]]:
        """Generate exactly n_requests arrivals when the scenario window permits."""

        if lam <= 0:
            raise ValueError("lam must be positive")
        if n_requests < 0:
            raise ValueError("n_requests must be non-negative")
        if not self.scenario.composition_schedule:
            return self._generate_fixed(lam, n_requests)
        return self._generate_scheduled(lam, n_requests)

    def generate_until(
        self, lam: float, duration_s: float
    ) -> list[tuple[float, Request]]:
        """Generate arrivals until duration_s using fixed or scheduled weights."""

        if lam <= 0:
            raise ValueError("lam must be positive")
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if not self.scenario.composition_schedule:
            return self._generate_fixed_until(lam, duration_s)
        return self._generate_scheduled_until(lam, duration_s)

    def _generate_fixed(
        self, lam: float, n_requests: int
    ) -> list[tuple[float, Request]]:
        arrivals: list[tuple[float, Request]] = []
        t = 0.0
        for req_id in range(n_requests):
            t += self._arrival_rng.expovariate(lam)
            archetype_id = self._choose_archetype(self.scenario.nominal_weights)
            arrivals.append((t, self._sample_request(archetype_id, req_id, t)))
        return arrivals

    def _generate_fixed_until(
        self, lam: float, duration_s: float
    ) -> list[tuple[float, Request]]:
        arrivals: list[tuple[float, Request]] = []
        t = 0.0
        while True:
            t += self._arrival_rng.expovariate(lam)
            if t > duration_s:
                return arrivals
            archetype_id = self._choose_archetype(self.scenario.nominal_weights)
            arrivals.append((t, self._sample_request(archetype_id, len(arrivals), t)))

    def _generate_scheduled(
        self, lam: float, n_requests: int
    ) -> list[tuple[float, Request]]:
        arrivals: list[tuple[float, Request]] = []
        for window in sorted(
            self.scenario.composition_schedule, key=lambda item: item.start_s
        ):
            self._append_window_arrivals(lam, window, arrivals, n_requests=n_requests)
            if len(arrivals) >= n_requests:
                return arrivals
        raise MixtureValidationError(
            "composition schedule ended before n_requests were generated; "
            "use generate_until or extend the schedule"
        )

    def _generate_scheduled_until(
        self,
        lam: float,
        duration_s: float,
    ) -> list[tuple[float, Request]]:
        arrivals: list[tuple[float, Request]] = []
        for window in sorted(
            self.scenario.composition_schedule, key=lambda item: item.start_s
        ):
            if window.start_s >= duration_s:
                break
            self._append_window_arrivals(
                lam,
                window,
                arrivals,
                duration_s=min(window.end_s, duration_s),
            )
        return arrivals

    def _append_window_arrivals(
        self,
        lam: float,
        window: CompositionWindow,
        arrivals: list[tuple[float, Request]],
        n_requests: int | None = None,
        duration_s: float | None = None,
    ) -> None:
        t = window.start_s
        end_s = duration_s if duration_s is not None else window.end_s
        effective_lam = lam * window.arrival_rate_multiplier
        while True:
            if n_requests is not None and len(arrivals) >= n_requests:
                return
            t += self._arrival_rng.expovariate(effective_lam)
            if t > end_s:
                return
            archetype_id = self._choose_archetype(window.weights)
            arrivals.append((t, self._sample_request(archetype_id, len(arrivals), t)))

    def _choose_archetype(self, weights: dict[str, float]) -> str:
        threshold = self._choice_rng.random()
        cumulative = 0.0
        last_id = ""
        for archetype_id, weight in weights.items():
            cumulative += weight
            last_id = archetype_id
            if threshold <= cumulative:
                return archetype_id
        return last_id

    def _sample_request(
        self, archetype_id: str, req_id: int, arrival: float
    ) -> Request:
        return self._samplers[archetype_id].sample_request(req_id, arrival)


class _ArchetypeSampler:
    def __init__(self, archetype: WorkloadArchetype, base_dir: Path | None, seed: int):
        self.archetype = archetype
        self._index = 0
        source = archetype.source
        if source.kind == "cdf":
            self._workload = CdfWorkload(
                _load_cdf_source(source, base_dir),
                l_in_frac=source.l_in_frac,
                l_out_frac=source.l_out_frac,
                category_mix=source.category_mix or None,
                seed=seed,
            )
            self._trace_requests: list[Request] | None = None
        elif source.kind == "trace":
            trace = TraceWorkload(
                str(source.resolve_path(base_dir)),
                fmt=source.format or "semantic_router",
                max_reqs=source.max_reqs,
                seed=seed,
                field_map=source.field_map,
                model_id_field=source.model_id_field,
                default_model_id=source.default_model_id,
            )
            self._trace_requests = [req for _, req in trace.generate()]
            if not self._trace_requests:
                raise MixtureValidationError(
                    f"{archetype.id}: trace source produced no requests"
                )
            self._workload = None
        else:
            raise MixtureValidationError(
                f"{archetype.id}: unsupported source kind {source.kind!r}"
            )

    def sample_request(self, req_id: int, arrival: float) -> Request:
        if self._trace_requests is not None:
            req = self._clone_trace_request(
                self._trace_requests[self._index], req_id, arrival
            )
            self._index = (self._index + 1) % len(self._trace_requests)
        else:
            req = self._workload.sample_request(req_id, arrival)
        req.archetype_id = self.archetype.id
        req.slo_class = self.archetype.slo_class
        req.eligible_models = self.archetype.model_eligibility
        req.residency = self.archetype.residency
        return req

    @staticmethod
    def _clone_trace_request(source: Request, req_id: int, arrival: float) -> Request:
        req = Request(
            req_id=req_id,
            arrival_time=arrival,
            l_in=source.l_in,
            l_out=source.l_out,
            category=source.category,
            model_id=source.model_id,
        )
        if hasattr(source, "_complexity"):
            req._complexity = source._complexity
        return req
