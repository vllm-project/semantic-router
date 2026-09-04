"""Evaluation validation and deterministic index materialization."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from catalog_common import (
    MIN_PIECEWISE_POINTS,
    PAIR_LENGTH,
    SHA256,
    VERSIONED_ID,
    CatalogBuildError,
)
from catalog_common import (
    is_finite_number as _is_finite_number,
)
from catalog_common import (
    mapping as _mapping,
)
from catalog_common import (
    nonempty_string as _nonempty_string,
)
from catalog_common import (
    reject_unknown as _reject_unknown,
)
from catalog_common import (
    sequence as _sequence,
)


def metric_catalog(benchmarks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for benchmark_index, benchmark in enumerate(benchmarks):
        path = f"benchmarks[{benchmark_index}]"
        _reject_unknown(
            benchmark, {"id", "display_name", "domain", "source", "metrics"}, path
        )
        benchmark_id = _nonempty_string(benchmark.get("id"), f"{path}.id")
        if not VERSIONED_ID.fullmatch(benchmark_id):
            raise CatalogBuildError(
                f"{path}.id must be a namespaced semantic-version identity"
            )
        for metric_index, raw_metric in enumerate(
            _sequence(benchmark.get("metrics"), f"{path}.metrics")
        ):
            metric = _mapping(raw_metric, f"{path}.metrics[{metric_index}]")
            _reject_unknown(
                metric,
                {"id", "unit", "direction", "range"},
                f"{path}.metrics[{metric_index}]",
            )
            metric_id = (
                f"{benchmark_id}#"
                f"{_nonempty_string(metric.get('id'), f'{path}.metrics[{metric_index}].id')}"
            )
            _validate_metric(metric, f"{path}.metrics[{metric_index}]")
            if metric_id in metrics:
                raise CatalogBuildError(f"duplicate benchmark metric: {metric_id}")
            metrics[metric_id] = {
                **metric,
                "benchmark": benchmark_id,
                "domain": benchmark.get("domain"),
            }
    return metrics


def _validate_metric(metric: dict[str, Any], path: str) -> None:
    value_range = _sequence(metric.get("range"), f"{path}.range")
    if (
        len(value_range) != PAIR_LENGTH
        or not all(_is_finite_number(value) for value in value_range)
        or value_range[0] >= value_range[1]
    ):
        raise CatalogBuildError(f"{path}.range is invalid")
    if metric.get("direction") not in {"higher_is_better", "lower_is_better"}:
        raise CatalogBuildError(f"{path}.direction is unsupported")


def validate_indices(
    items: list[dict[str, Any]], metrics: dict[str, dict[str, Any]]
) -> None:
    index_ids = {item.get("id") for item in items}
    edges: dict[str, set[str]] = defaultdict(set)
    for index, item in enumerate(items):
        path = f"indices[{index}]"
        identity, domains = _validate_index_header(item, path)
        total, direct_domain_weights, has_nested_component = _validate_index_components(
            item, path, identity, index_ids, metrics, edges
        )
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise CatalogBuildError(
                f"{path}.component weights must sum to 1, got {total}"
            )
        _validate_direct_domain_weights(
            path, domains, direct_domain_weights, has_nested_component
        )
    _validate_index_cycles(
        {identity for identity in index_ids if isinstance(identity, str)}, edges
    )


def _validate_direct_domain_weights(
    path: str,
    domains: dict[str, Any],
    direct_domain_weights: dict[str, float],
    has_nested_component: bool,
) -> None:
    if has_nested_component:
        return
    matches = set(direct_domain_weights) == set(domains) and all(
        math.isclose(direct_domain_weights[domain], float(weight), abs_tol=1e-9)
        for domain, weight in domains.items()
    )
    if not matches:
        raise CatalogBuildError(f"{path}.domains do not match direct component weights")


_INDEX_FIELDS = {
    "id",
    "display_name",
    "description",
    "methodology",
    "aggregation",
    "scale",
    "missing",
    "domains",
    "components",
}
_NORMALIZATION_FIELDS = {"type", "min", "max", "k", "x0", "points", "values"}
_NORMALIZATION_TYPES = {
    "identity",
    "one_minus",
    "linear_clamp",
    "piecewise_linear",
    "logistic",
    "lookup",
}


def _validate_index_header(
    item: dict[str, Any], path: str
) -> tuple[str, dict[str, Any]]:
    _reject_unknown(item, _INDEX_FIELDS, path)
    identity = _nonempty_string(item.get("id"), f"{path}.id")
    if not VERSIONED_ID.fullmatch(identity):
        raise CatalogBuildError(
            f"{path}.id must be a namespaced semantic-version identity"
        )
    if item.get("aggregation") != "weighted_mean":
        raise CatalogBuildError(f"{path}.aggregation is unsupported")
    _validate_index_scale(item, path)
    _validate_index_missing_policy(item, path)
    return identity, _validate_index_domains(item, path)


def _validate_index_scale(item: dict[str, Any], path: str) -> None:
    scale = _sequence(item.get("scale"), f"{path}.scale")
    if (
        len(scale) != PAIR_LENGTH
        or not all(_is_finite_number(value) for value in scale)
        or scale[0] >= scale[1]
    ):
        raise CatalogBuildError(f"{path}.scale is invalid")


def _validate_index_missing_policy(item: dict[str, Any], path: str) -> None:
    missing = _mapping(item.get("missing"), f"{path}.missing")
    _reject_unknown(missing, {"policy", "minimum"}, f"{path}.missing")
    policy = missing.get("policy")
    if policy not in {"require_all", "require_coverage", "reported_only"}:
        raise CatalogBuildError(f"{path}.missing.policy is unsupported")
    if policy == "require_coverage" and not _is_finite_number(missing.get("minimum")):
        raise CatalogBuildError(f"{path}.missing.minimum is required")


def _validate_index_domains(item: dict[str, Any], path: str) -> dict[str, Any]:
    domains = _mapping(item.get("domains"), f"{path}.domains")
    if not domains or any(
        not _is_finite_number(weight) or weight <= 0 for weight in domains.values()
    ):
        raise CatalogBuildError(f"{path}.domains must contain positive finite weights")
    total = sum(float(weight) for weight in domains.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise CatalogBuildError(f"{path}.domain weights must sum to 1, got {total}")
    return domains


def _validate_index_components(
    item: dict[str, Any],
    path: str,
    identity: str,
    index_ids: set[Any],
    metrics: dict[str, dict[str, Any]],
    edges: dict[str, set[str]],
) -> tuple[float, dict[str, float], bool]:
    components = _sequence(item.get("components"), f"{path}.components")
    if not components:
        raise CatalogBuildError(f"{path}.components cannot be empty")
    total = 0.0
    direct_domains: dict[str, float] = defaultdict(float)
    nested = False
    for component_index, raw_component in enumerate(components):
        component_path = f"{path}.components[{component_index}]"
        weight, domain, dependency = _validate_index_component(
            raw_component, component_path, index_ids, metrics
        )
        total += weight
        if domain is not None:
            direct_domains[domain] += weight
        if dependency is not None:
            nested = True
            edges[identity].add(dependency)
    return total, direct_domains, nested


def _validate_index_component(
    raw_component: Any,
    path: str,
    index_ids: set[Any],
    metrics: dict[str, dict[str, Any]],
) -> tuple[float, str | None, str | None]:
    component = _mapping(raw_component, path)
    _reject_unknown(component, {"metric", "index", "weight", "normalization"}, path)
    references = [key for key in ("metric", "index") if component.get(key)]
    if len(references) != 1:
        raise CatalogBuildError(f"{path} must reference exactly one metric or index")
    metric_id = component.get("metric")
    dependency = component.get("index")
    if metric_id and metric_id not in metrics:
        raise CatalogBuildError(f"{path} references an unknown metric")
    if dependency and dependency not in index_ids:
        raise CatalogBuildError(f"{path} references an unknown index")
    weight = component.get("weight")
    if not _is_finite_number(weight) or weight <= 0:
        raise CatalogBuildError(f"{path}.weight must be positive")
    normalization_path = f"{path}.normalization"
    normalization = _mapping(
        component.get("normalization", {"type": "identity"}), normalization_path
    )
    _validate_normalization(normalization, normalization_path)
    domain = str(metrics[metric_id]["domain"]) if metric_id else None
    return float(weight), domain, str(dependency) if dependency else None


def _validate_normalization(normalization: dict[str, Any], path: str) -> None:
    _reject_unknown(normalization, _NORMALIZATION_FIELDS, path)
    kind = normalization.get("type")
    if kind not in _NORMALIZATION_TYPES:
        raise CatalogBuildError(f"{path}.type is unsupported")
    if kind == "linear_clamp":
        _validate_linear_clamp(normalization, path)
    elif kind == "piecewise_linear":
        _validate_piecewise_points(normalization.get("points"), path)
    elif kind == "logistic":
        _validate_logistic(normalization, path)
    elif kind == "lookup":
        _validate_lookup(normalization.get("values"), path)


def _validate_linear_clamp(normalization: dict[str, Any], path: str) -> None:
    minimum, maximum = normalization.get("min"), normalization.get("max")
    if (
        not _is_finite_number(minimum)
        or not _is_finite_number(maximum)
        or minimum >= maximum
    ):
        raise CatalogBuildError(f"{path} bounds are invalid")


def _validate_piecewise_points(points: Any, path: str) -> None:
    if not isinstance(points, list) or len(points) < MIN_PIECEWISE_POINTS:
        raise CatalogBuildError(f"{path}.points requires at least two entries")
    previous_input: float | None = None
    for point_index, point in enumerate(points):
        point_path = f"{path}.points[{point_index}]"
        if not isinstance(point, dict) or set(point) != {"input", "output"}:
            raise CatalogBuildError(f"{point_path} is invalid")
        point_input, point_output = point["input"], point["output"]
        if not _is_finite_number(point_input) or not _is_finite_number(point_output):
            raise CatalogBuildError(f"{point_path} must be finite")
        if not 0 <= point_output <= 1:
            raise CatalogBuildError(
                f"{path}.points must have increasing inputs and outputs in [0, 1]"
            )
        if previous_input is not None and point_input <= previous_input:
            raise CatalogBuildError(
                f"{path}.points must have increasing inputs and outputs in [0, 1]"
            )
        previous_input = float(point_input)


def _validate_logistic(normalization: dict[str, Any], path: str) -> None:
    k, x0 = normalization.get("k"), normalization.get("x0")
    if not _is_finite_number(k) or not _is_finite_number(x0) or k == 0:
        raise CatalogBuildError(f"{path} requires finite non-zero k and finite x0")


def _validate_lookup(values: Any, path: str) -> None:
    if not isinstance(values, dict) or not values:
        raise CatalogBuildError(f"{path}.values is invalid")
    if any(
        not isinstance(key, str)
        or not key
        or not _is_finite_number(value)
        or not 0 <= value <= 1
        for key, value in values.items()
    ):
        raise CatalogBuildError(f"{path}.values is invalid")


def _validate_index_cycles(index_ids: set[str], edges: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identity: str) -> None:
        if identity in visiting:
            raise CatalogBuildError(f"index dependency cycle includes {identity}")
        if identity in visited:
            return
        visiting.add(identity)
        for dependency in edges.get(identity, set()):
            visit(dependency)
        visiting.remove(identity)
        visited.add(identity)

    for identity in sorted(index_ids):
        visit(identity)


def validate_offerings(
    items: list[dict[str, Any]],
    providers: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    protocol_ids: set[str],
) -> None:
    offered_models: set[str] = set()
    for index, item in enumerate(items):
        path = f"offerings[{index}]"
        _reject_unknown(
            item,
            {
                "id",
                "provider",
                "model",
                "provider_model_id",
                "protocols",
                "pricing",
                "restrictions",
                "lifecycle",
                "verification",
            },
            path,
        )
        provider = providers.get(str(item.get("provider")))
        model = models.get(str(item.get("model")))
        if provider is None or model is None:
            raise CatalogBuildError(f"{path} references an unknown provider or model")
        protocols = _sequence(item.get("protocols"), f"{path}.protocols")
        if not protocols or any(protocol not in protocol_ids for protocol in protocols):
            raise CatalogBuildError(f"{path}.protocols references an unknown protocol")
        if any(protocol not in provider["protocols"] for protocol in protocols):
            raise CatalogBuildError(
                f"{path}.protocols is not supported by its provider"
            )
        if any(
            f"{protocol}#create" not in provider["supported_operations"]
            for protocol in protocols
        ):
            raise CatalogBuildError(
                f"{path}.protocols has no provider create operation"
            )
        if any(protocol not in model["protocols"] for protocol in protocols):
            raise CatalogBuildError(f"{path}.protocols is not supported by its model")
        offered_models.add(str(item["model"]))
    missing = sorted(
        model_id
        for model_id, model in models.items()
        if model.get("kind") == "physical"
        and model.get("lifecycle") != "removed"
        and model_id not in offered_models
    )
    if missing:
        raise CatalogBuildError(
            "physical models require at least one provider offering: "
            + ", ".join(missing)
        )


def validate_evaluations(
    items: list[dict[str, Any]],
    model_ids: set[str],
    metrics: dict[str, dict[str, Any]],
) -> None:
    available_metrics: dict[tuple[str, str], str] = {}
    for index, item in enumerate(items):
        path = f"evaluations[{index}]"
        _reject_unknown(
            item,
            {"id", "model", "subject", "metrics", "status", "measured_at", "evidence"},
            path,
        )
        if item.get("model") not in model_ids:
            raise CatalogBuildError(f"{path}.model references an unknown model")
        if item.get("status") not in {
            "available",
            "missing",
            "failed",
            "not_applicable",
            "withheld",
        }:
            raise CatalogBuildError(f"{path}.status is unsupported")
        values = _mapping(item.get("metrics", {}), f"{path}.metrics")
        _validate_evaluation_values(values, path, metrics)
        _validate_evaluation_evidence(item, values, path)
        if item.get("status") == "available":
            for metric_id in values:
                key = (str(item["model"]), metric_id)
                previous = available_metrics.get(key)
                if previous is not None:
                    raise CatalogBuildError(
                        f"{path} conflicts with {previous}: one available value is "
                        f"allowed per model and benchmark metric"
                    )
                available_metrics[key] = path


def _validate_evaluation_values(
    values: dict[str, Any], path: str, metrics: dict[str, dict[str, Any]]
) -> None:
    for metric_id, value in values.items():
        definition = metrics.get(metric_id)
        if definition is None:
            raise CatalogBuildError(
                f"{path}.metrics references an unknown metric: {metric_id}"
            )
        if not _is_finite_number(value):
            raise CatalogBuildError(f"{path}.metrics.{metric_id} must be finite")
        if value < definition["range"][0] or value > definition["range"][1]:
            raise CatalogBuildError(
                f"{path}.metrics.{metric_id} is outside its declared range"
            )


def _validate_evaluation_evidence(
    item: dict[str, Any], values: dict[str, Any], path: str
) -> None:
    evidence = _mapping(item.get("evidence"), f"{path}.evidence")
    _reject_unknown(
        evidence,
        {"provenance", "verification", "source", "artifact", "redistributable"},
        f"{path}.evidence",
    )
    if item.get("status") == "available" and not values:
        raise CatalogBuildError(
            f"{path}.metrics cannot be empty for an available record"
        )
    if (
        item.get("status") == "available"
        and evidence.get("redistributable") is not True
    ):
        raise CatalogBuildError(
            f"{path} cannot be published without redistribution permission"
        )
    artifact = evidence.get("artifact")
    if artifact and not SHA256.fullmatch(str(artifact)):
        raise CatalogBuildError(f"{path}.evidence.artifact must be a SHA-256 digest")


def normalize_component(value: float, normalization: dict[str, Any]) -> float:
    kind = normalization.get("type", "identity")
    if kind == "identity":
        return min(1.0, max(0.0, value))
    if kind == "one_minus":
        return min(1.0, max(0.0, 1.0 - value))
    if kind == "linear_clamp":
        minimum = float(normalization["min"])
        maximum = float(normalization["max"])
        return min(1.0, max(0.0, (value - minimum) / (maximum - minimum)))
    if kind == "piecewise_linear":
        return _piecewise_normalization(value, normalization["points"])
    if kind == "logistic":
        return 1.0 / (
            1.0
            + math.exp(
                -float(normalization["k"]) * (value - float(normalization["x0"]))
            )
        )
    if kind == "lookup":
        return _lookup_normalization(value, normalization["values"])
    raise CatalogBuildError(f"unsupported normalization: {kind}")


def _piecewise_normalization(value: float, points: list[dict[str, Any]]) -> float:
    if value <= points[0]["input"]:
        return float(points[0]["output"])
    for left, right in pairwise(points):
        if value <= right["input"]:
            ratio = (value - left["input"]) / (right["input"] - left["input"])
            return float(left["output"] + ratio * (right["output"] - left["output"]))
    return float(points[-1]["output"])


def _lookup_normalization(value: float, values: dict[str, Any]) -> float:
    key = format(value, "g")
    try:
        return float(values[key])
    except KeyError as error:
        raise CatalogBuildError(
            f"lookup normalization has no entry for {key}"
        ) from error


@dataclass
class _IndexAccumulation:
    weighted: float = 0.0
    present_weight: float = 0.0
    components: list[dict[str, Any]] = field(default_factory=list)
    provenance: set[str] = field(default_factory=set)
    domain_weighted: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    domain_coverage: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )


class _IndexEvaluator:
    def __init__(
        self,
        model: dict[str, Any],
        measurements: dict[str, tuple[float, dict[str, Any]]],
        definitions: dict[str, dict[str, Any]],
        benchmarks: dict[str, dict[str, Any]],
    ) -> None:
        self.model = model
        self.measurements = measurements
        self.definitions = definitions
        self.benchmarks = benchmarks
        self.memo: dict[str, dict[str, Any]] = {}

    def compute(self, index_id: str, visiting: set[str]) -> dict[str, Any]:
        if index_id in self.memo:
            return self.memo[index_id]
        if index_id in visiting:
            raise CatalogBuildError(f"index dependency cycle includes {index_id}")
        if self.model.get("kind") == "virtual":
            return self._store(index_id, self._not_applicable(index_id))
        visiting.add(index_id)
        definition = self.definitions[index_id]
        accumulation = self._accumulate(definition, visiting)
        visiting.remove(index_id)
        return self._store(index_id, self._result(index_id, definition, accumulation))

    def _accumulate(
        self, definition: dict[str, Any], visiting: set[str]
    ) -> _IndexAccumulation:
        accumulation = _IndexAccumulation()
        for component in definition["components"]:
            value, domain, provenance = self._component_value(component, visiting)
            accumulation.provenance.update(provenance)
            component_result = _missing_component_result(component)
            if value is None:
                accumulation.components.append(component_result)
                continue
            normalized = normalize_component(
                value, component.get("normalization", {"type": "identity"})
            )
            _accumulate_component(
                accumulation, component, component_result, value, normalized, domain
            )
        return accumulation

    def _component_value(
        self, component: dict[str, Any], visiting: set[str]
    ) -> tuple[float | None, str | None, set[str]]:
        metric_id = component.get("metric")
        if metric_id:
            measurement = self.measurements.get(metric_id)
            if measurement is None:
                return None, None, set()
            value, record = measurement
            benchmark_id = metric_id.split("#", 1)[0]
            return (
                value,
                str(self.benchmarks[benchmark_id]["domain"]),
                {str(record["id"])},
            )
        dependency_id = str(component["index"])
        dependency = self.compute(dependency_id, visiting)
        provenance = set(dependency["provenance"])
        if dependency["status"] != "available" or dependency["score"] is None:
            return None, None, provenance
        lower, upper = self.definitions[dependency_id]["scale"]
        value = (dependency["score"] - lower) / (upper - lower)
        return float(value), None, provenance

    def _result(
        self,
        index_id: str,
        definition: dict[str, Any],
        accumulation: _IndexAccumulation,
    ) -> dict[str, Any]:
        available = _index_is_available(definition, accumulation.present_weight)
        result: dict[str, Any] = {
            "model": self.model["id"],
            "index": index_id,
            "status": "available" if available else "missing",
            "score": _index_score(definition, accumulation, available),
            "coverage": accumulation.present_weight,
            "components": accumulation.components,
            "provenance": sorted(accumulation.provenance),
        }
        domains = _index_domain_scores(definition, accumulation, available)
        if domains:
            result["domains"] = domains
        return result

    def _not_applicable(self, index_id: str) -> dict[str, Any]:
        return {
            "model": self.model["id"],
            "index": index_id,
            "status": "not_applicable",
            "score": None,
            "coverage": 0.0,
            "components": [],
            "provenance": [],
        }

    def _store(self, index_id: str, result: dict[str, Any]) -> dict[str, Any]:
        self.memo[index_id] = result
        return result


def _accumulate_component(
    accumulation: _IndexAccumulation,
    component: dict[str, Any],
    result: dict[str, Any],
    value: float,
    normalized: float,
    domain: str | None,
) -> None:
    weight = float(component["weight"])
    accumulation.weighted += weight * normalized
    accumulation.present_weight += weight
    if domain is not None:
        accumulation.domain_weighted[domain] += weight * normalized
        accumulation.domain_coverage[domain] += weight
    result.update(
        {
            "weight": weight,
            "status": "available",
            "value": value,
            "normalized": normalized,
        }
    )
    accumulation.components.append(result)


def _missing_component_result(component: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "weight": component["weight"],
        "status": "missing",
        "value": None,
        "normalized": None,
    }
    if component.get("metric"):
        result["metric"] = component["metric"]
    if component.get("index"):
        result["index"] = component["index"]
    return result


def _index_is_available(definition: dict[str, Any], present_weight: float) -> bool:
    if present_weight <= 0:
        return False
    policy = definition["missing"]["policy"]
    if policy == "require_all":
        return math.isclose(present_weight, 1.0, abs_tol=1e-9)
    if policy == "require_coverage":
        return present_weight >= float(definition["missing"]["minimum"])
    return policy == "reported_only"


def _index_score(
    definition: dict[str, Any], accumulation: _IndexAccumulation, available: bool
) -> float | None:
    if not available:
        return None
    policy = definition["missing"]["policy"]
    normalized = accumulation.weighted
    if policy != "require_all":
        normalized /= accumulation.present_weight
    lower, upper = definition["scale"]
    return float(lower + normalized * (upper - lower))


def _index_domain_scores(
    definition: dict[str, Any], accumulation: _IndexAccumulation, available: bool
) -> dict[str, float]:
    if not available:
        return {}
    lower, upper = definition["scale"]
    return {
        domain: float(
            lower + (accumulation.domain_weighted[domain] / coverage) * (upper - lower)
        )
        for domain, coverage in sorted(accumulation.domain_coverage.items())
    }


def _available_evaluations(
    evaluations: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float, dict[str, Any]]]]:
    records_by_model: dict[str, dict[str, tuple[float, dict[str, Any]]]] = defaultdict(
        dict
    )
    for record in evaluations:
        if record.get("status") != "available":
            continue
        for metric_id, value in record.get("metrics", {}).items():
            records_by_model[record["model"]][metric_id] = (float(value), record)
    return records_by_model


def index_results(resources: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    measurements = _available_evaluations(resources["evaluations"])
    definitions = {definition["id"]: definition for definition in resources["indices"]}
    benchmarks = {
        definition["id"]: definition for definition in resources["benchmarks"]
    }
    results: list[dict[str, Any]] = []
    for model in resources["models"]:
        evaluator = _IndexEvaluator(
            model, measurements.get(model["id"], {}), definitions, benchmarks
        )
        results.extend(
            evaluator.compute(definition["id"], set())
            for definition in resources["indices"]
        )
    return results
