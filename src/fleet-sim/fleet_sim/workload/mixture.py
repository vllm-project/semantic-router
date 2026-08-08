"""Versioned workload archetype mixture scenarios.

Mixture scenarios let FleetSim build deterministic request streams from named
workload archetypes instead of collapsing all demand into one aggregate CDF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "fleet-sim.workload-mixture/v1alpha1"
WEIGHT_TOLERANCE = 1e-6


class MixtureValidationError(ValueError):
    """Raised when a workload mixture scenario fails validation."""


@dataclass(frozen=True)
class ArchetypeSource:
    """Input source for an archetype's request token distribution."""

    kind: str
    path: str
    format: str | None = None
    field_map: dict[str, str] = field(default_factory=dict)
    model_id_field: str | None = None
    default_model_id: str | None = None
    max_reqs: int | None = None
    l_in_frac: float = 0.80
    l_out_frac: float = 0.20
    category_mix: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchetypeSource:
        return cls(
            kind=str(data.get("kind", "")),
            path=str(data.get("path", "")),
            format=data.get("format"),
            field_map=dict(data.get("field_map") or {}),
            model_id_field=data.get("model_id_field"),
            default_model_id=data.get("default_model_id"),
            max_reqs=data.get("max_reqs"),
            l_in_frac=float(data.get("l_in_frac", 0.80)),
            l_out_frac=float(data.get("l_out_frac", 0.20)),
            category_mix=dict(data.get("category_mix") or {}),
        )

    def resolve_path(self, base_dir: Path | None) -> Path:
        path = Path(self.path)
        if path.is_absolute() or base_dir is None:
            return path
        return base_dir / path


@dataclass(frozen=True)
class WorkloadArchetype:
    """Versioned workload archetype in a mixture scenario."""

    id: str
    version: str
    source: ArchetypeSource
    arrival_process: dict[str, Any] = field(default_factory=dict)
    slo_class: str | None = None
    model_eligibility: tuple[str, ...] = ()
    residency: tuple[str, ...] = ()
    weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadArchetype:
        return cls(
            id=str(data.get("id", "")),
            version=str(data.get("version", "")),
            source=ArchetypeSource.from_dict(dict(data.get("source") or {})),
            arrival_process=dict(data.get("arrival_process") or {"kind": "poisson"}),
            slo_class=data.get("slo_class"),
            model_eligibility=tuple(str(v) for v in data.get("model_eligibility", ())),
            residency=tuple(str(v) for v in data.get("residency", ())),
            weight=float(data.get("weight", 0.0)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CompositionWindow:
    """A time window with its own archetype weights and arrival multiplier."""

    start_s: float
    duration_s: float
    weights: dict[str, float]
    arrival_rate_multiplier: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompositionWindow:
        return cls(
            start_s=float(data.get("start_s", 0.0)),
            duration_s=float(data.get("duration_s", 0.0)),
            weights={
                str(k): float(v) for k, v in dict(data.get("weights") or {}).items()
            },
            arrival_rate_multiplier=float(data.get("arrival_rate_multiplier", 1.0)),
        )

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class MixtureScenario:
    """Versioned workload archetype mixture scenario."""

    id: str
    version: str
    archetypes: tuple[WorkloadArchetype, ...]
    schema_version: str = SCHEMA_VERSION
    seed: int = 42
    composition_schedule: tuple[CompositionWindow, ...] = ()
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    base_dir: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        base_dir: str | Path | None = None,
    ) -> MixtureScenario:
        return cls(
            schema_version=str(data.get("schema_version", "")),
            id=str(data.get("id", "")),
            version=str(data.get("version", "")),
            seed=int(data.get("seed", 42)),
            description=data.get("description"),
            archetypes=tuple(
                WorkloadArchetype.from_dict(dict(item))
                for item in data.get("archetypes", ())
            ),
            composition_schedule=tuple(
                CompositionWindow.from_dict(dict(item))
                for item in data.get("composition_schedule", ())
            ),
            metadata=dict(data.get("metadata") or {}),
            base_dir=Path(base_dir) if base_dir is not None else None,
        )

    @property
    def archetype_ids(self) -> tuple[str, ...]:
        return tuple(archetype.id for archetype in self.archetypes)

    @property
    def archetype_map(self) -> dict[str, WorkloadArchetype]:
        return {archetype.id: archetype for archetype in self.archetypes}

    @property
    def nominal_weights(self) -> dict[str, float]:
        return {archetype.id: archetype.weight for archetype in self.archetypes}


def load_mixture_scenario(path: str | Path, validate: bool = True) -> MixtureScenario:
    """Load a workload mixture scenario JSON file."""

    scenario_path = Path(path)
    data = json.loads(scenario_path.read_text())
    scenario = MixtureScenario.from_dict(data, base_dir=scenario_path.parent)
    if validate:
        validate_mixture_scenario(scenario)
    return scenario


def validate_cdf_points(cdf: list[Any], name: str = "cdf") -> list[str]:
    """Return validation errors for a total-token empirical CDF."""

    errors: list[str] = []
    if not cdf:
        return [f"{name}: CDF must not be empty"]

    prev_threshold = 0
    prev_fraction = 0.0
    parsed: list[tuple[int, float]] = []
    for idx, point in enumerate(cdf):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            errors.append(
                f"{name}: point {idx} must be [token_length, cumulative_fraction]"
            )
            continue
        try:
            threshold = int(point[0])
            fraction = float(point[1])
        except (TypeError, ValueError):
            errors.append(f"{name}: point {idx} contains non-numeric values")
            continue
        if threshold <= 0:
            errors.append(f"{name}: point {idx} token_length must be positive")
        if threshold <= prev_threshold:
            errors.append(
                f"{name}: point {idx} token_length must be strictly increasing"
            )
        if not 0.0 <= fraction <= 1.0:
            errors.append(f"{name}: point {idx} cumulative_fraction must be in [0, 1]")
        if fraction < prev_fraction:
            errors.append(
                f"{name}: point {idx} cumulative_fraction must be non-decreasing"
            )
        prev_threshold = threshold
        prev_fraction = fraction
        parsed.append((threshold, fraction))

    if parsed and abs(parsed[-1][1] - 1.0) > WEIGHT_TOLERANCE:
        errors.append(f"{name}: final cumulative_fraction must be 1.0")
    return errors


def validate_mixture_scenario(scenario: MixtureScenario) -> None:
    """Validate a scenario and raise MixtureValidationError on any error."""

    errors: list[str] = []
    if scenario.schema_version != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}, got {scenario.schema_version!r}"
        )
    if not scenario.id:
        errors.append("scenario id is required")
    if not scenario.version:
        errors.append("scenario version is required")
    if not scenario.archetypes:
        errors.append("scenario must define at least one archetype")

    ids = scenario.archetype_ids
    duplicate_ids = sorted(
        {archetype_id for archetype_id in ids if ids.count(archetype_id) > 1}
    )
    if duplicate_ids:
        errors.append(f"archetype ids must be unique: {duplicate_ids}")

    id_set = set(ids)
    _validate_weight_map(scenario.nominal_weights, id_set, "nominal weights", errors)

    for archetype in scenario.archetypes:
        errors.extend(_validate_archetype(archetype, scenario.base_dir))

    last_end = None
    for index, window in enumerate(
        sorted(scenario.composition_schedule, key=lambda w: w.start_s)
    ):
        label = f"composition_schedule[{index}]"
        if window.start_s < 0:
            errors.append(f"{label}: start_s must be non-negative")
        if window.duration_s <= 0:
            errors.append(f"{label}: duration_s must be positive")
        if window.arrival_rate_multiplier <= 0:
            errors.append(f"{label}: arrival_rate_multiplier must be positive")
        if last_end is not None and window.start_s < last_end:
            errors.append(f"{label}: windows must not overlap")
        last_end = window.end_s
        _validate_weight_map(window.weights, id_set, f"{label} weights", errors)

    if errors:
        raise MixtureValidationError("\n".join(errors))


def _validate_archetype(
    archetype: WorkloadArchetype, base_dir: Path | None
) -> list[str]:
    errors: list[str] = []
    label = f"archetype {archetype.id!r}"
    if not archetype.id:
        errors.append("archetype id is required")
    if not archetype.version:
        errors.append(f"{label}: version is required")
    if archetype.source.kind not in {"cdf", "trace"}:
        errors.append(f"{label}: source.kind must be 'cdf' or 'trace'")
    if archetype.weight < 0:
        errors.append(f"{label}: weight must be non-negative")
    arrival_kind = archetype.arrival_process.get("kind", "poisson")
    if arrival_kind != "poisson":
        errors.append(f"{label}: only poisson arrival_process is supported in v1alpha1")
    if archetype.source.kind == "cdf":
        try:
            cdf = _load_cdf_source(archetype.source, base_dir)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: failed to load CDF source: {exc}")
        else:
            errors.extend(validate_cdf_points(cdf, f"{label} source CDF"))
    elif archetype.source.kind == "trace":
        path = archetype.source.resolve_path(base_dir)
        if not path.exists():
            errors.append(f"{label}: trace source does not exist: {path}")
    return errors


def _validate_weight_map(
    weights: dict[str, float],
    expected_ids: set[str],
    label: str,
    errors: list[str],
) -> None:
    actual_ids = set(weights)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing:
        errors.append(f"{label}: missing weights for {missing}")
    if extra:
        errors.append(f"{label}: unknown archetype ids {extra}")
    for archetype_id, weight in weights.items():
        if weight < 0:
            errors.append(f"{label}: {archetype_id} weight must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        errors.append(f"{label}: weights must sum to 1.0, got {total:.6f}")


def _load_cdf_source(
    source: ArchetypeSource,
    base_dir: Path | None,
) -> list[tuple[int, float]]:
    raw = json.loads(source.resolve_path(base_dir).read_text())
    cdf = raw["cdf"] if isinstance(raw, dict) else raw
    return [(int(threshold), float(fraction)) for threshold, fraction in cdf]
