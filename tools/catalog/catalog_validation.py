"""Cross-resource validation for provider bindings and evaluation records."""

from __future__ import annotations

from typing import Any

from catalog_common import SHA256, SLUG, CatalogBuildError
from catalog_common import is_finite_number as _is_finite_number
from catalog_common import mapping as _mapping
from catalog_common import nonempty_string as _nonempty_string
from catalog_common import reject_unknown as _reject_unknown
from catalog_common import sequence as _sequence

REASONING_TRANSPORTS = {
    "chat_template_kwargs",
    "top_level_effort",
    "top_level_boolean",
    "reasoning_object",
    "thinking_object",
    "deepseek_thinking",
}


def validate_provider_bindings(
    providers: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    protocol_ids: set[str],
) -> None:
    bound_models: set[str] = set()
    for provider_id, provider in providers.items():
        native_ids: set[str] = set()
        pairs: set[tuple[str, str]] = set()
        for index, item in enumerate(provider.get("models", [])):
            path = f"providers[{provider_id}].models[{index}]"
            _validate_provider_binding(
                item, path, provider, models, protocol_ids, native_ids, pairs
            )
            bound_models.add(str(item["catalog"]))
    _validate_every_physical_model_is_bound(models, bound_models)


def _validate_provider_binding(
    item: dict[str, Any],
    path: str,
    provider: dict[str, Any],
    models: dict[str, dict[str, Any]],
    protocol_ids: set[str],
    native_ids: set[str],
    pairs: set[tuple[str, str]],
) -> None:
    _reject_unknown(
        item,
        {
            "catalog",
            "id",
            "protocols",
            "reasoning_transport",
            "pricing",
            "restrictions",
            "lifecycle",
            "verification",
        },
        path,
    )
    model_id = _nonempty_string(item.get("catalog"), f"{path}.catalog")
    native_id = _nonempty_string(item.get("id"), f"{path}.id")
    if (
        item.get("reasoning_transport", "chat_template_kwargs")
        not in REASONING_TRANSPORTS
    ):
        raise CatalogBuildError(f"{path}.reasoning_transport is unsupported")
    if model_id not in models:
        raise CatalogBuildError(f"{path}.catalog references an unknown model")
    if native_id in native_ids:
        raise CatalogBuildError(
            f"{path}.id duplicates a provider-native model identifier"
        )
    native_ids.add(native_id)
    protocols = _sequence(item.get("protocols"), f"{path}.protocols")
    _validate_binding_protocols(protocols, path, provider, protocol_ids)
    for protocol in protocols:
        pair = (model_id, str(protocol))
        if pair in pairs:
            raise CatalogBuildError(
                f"{path} duplicates catalog model {model_id} for {protocol}"
            )
        pairs.add(pair)


def _validate_binding_protocols(
    protocols: list[Any],
    path: str,
    provider: dict[str, Any],
    protocol_ids: set[str],
) -> None:
    if not protocols or any(protocol not in protocol_ids for protocol in protocols):
        raise CatalogBuildError(f"{path}.protocols references an unknown protocol")
    if any(protocol not in provider["protocols"] for protocol in protocols):
        raise CatalogBuildError(f"{path}.protocols is not supported by its provider")
    if any(
        f"{protocol}#create" not in provider["supported_operations"]
        for protocol in protocols
    ):
        raise CatalogBuildError(f"{path}.protocols has no provider create operation")


def _validate_every_physical_model_is_bound(
    models: dict[str, dict[str, Any]], bound_models: set[str]
) -> None:
    missing = sorted(
        model_id
        for model_id, model in models.items()
        if model.get("kind") == "physical"
        and model.get("lifecycle") != "removed"
        and model_id not in bound_models
    )
    if missing:
        raise CatalogBuildError(
            "physical models require at least one provider binding: "
            + ", ".join(missing)
        )


def validate_evaluations(
    items: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    reasoning_families: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> None:
    available_metrics: dict[tuple[str, str, str, str, str], str] = {}
    for index, item in enumerate(items):
        path = f"evaluations[{index}]"
        model_id, effort, benchmark, profile, values = _validate_evaluation(
            item, path, models, reasoning_families, metrics
        )
        if item.get("status") != "available":
            continue
        for metric_id in values:
            key = (model_id, effort, benchmark, profile, metric_id)
            previous = available_metrics.get(key)
            if previous is not None:
                raise CatalogBuildError(
                    f"{path} conflicts with {previous}: one available value is "
                    "allowed per model, reasoning effort, benchmark profile, and metric"
                )
            available_metrics[key] = path


def _validate_evaluation(
    item: dict[str, Any],
    path: str,
    models: dict[str, dict[str, Any]],
    reasoning_families: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> tuple[str, str, str, str, dict[str, Any]]:
    _reject_unknown(
        item,
        {
            "id",
            "model",
            "benchmark",
            "benchmark_profile",
            "reasoning_effort",
            "subject",
            "metrics",
            "status",
            "measured_at",
            "evidence",
        },
        path,
    )
    model_id = str(item.get("model"))
    if model_id not in models:
        raise CatalogBuildError(f"{path}.model references an unknown model")
    if item.get("status") not in {
        "available",
        "missing",
        "failed",
        "not_applicable",
        "withheld",
    }:
        raise CatalogBuildError(f"{path}.status is unsupported")
    benchmark = _nonempty_string(item.get("benchmark"), f"{path}.benchmark")
    profile = _nonempty_string(
        item.get("benchmark_profile"), f"{path}.benchmark_profile"
    )
    effort = _nonempty_string(item.get("reasoning_effort"), f"{path}.reasoning_effort")
    if not SLUG.fullmatch(effort):
        raise CatalogBuildError(f"{path}.reasoning_effort must be a slug")
    _validate_reasoning_effort(
        effort, models[model_id], reasoning_families, f"{path}.reasoning_effort"
    )
    values = _mapping(item.get("metrics", {}), f"{path}.metrics")
    _validate_evaluation_values(values, benchmark, profile, path, metrics)
    _validate_evaluation_evidence(item, values, path)
    return model_id, effort, benchmark, profile, values


def _validate_reasoning_effort(
    effort: str,
    model: dict[str, Any],
    reasoning_families: dict[str, dict[str, Any]],
    path: str,
) -> None:
    family_id = model.get("reasoning_family")
    if family_id is None:
        return
    family = reasoning_families.get(str(family_id))
    if family is None:
        raise CatalogBuildError(f"{path} references an unknown reasoning family")
    if effort not in {*family["levels"], "published"}:
        raise CatalogBuildError(
            f"{path} {effort!r} is not supported by model {model['id']}"
        )


def _validate_evaluation_values(
    values: dict[str, Any],
    benchmark: str,
    profile: str,
    path: str,
    metrics: dict[str, dict[str, Any]],
) -> None:
    for metric_id, value in values.items():
        definition = metrics.get(f"{benchmark}#{metric_id}")
        if definition is None:
            raise CatalogBuildError(
                f"{path}.metrics references an unknown metric for {benchmark}: {metric_id}"
            )
        if profile not in definition["profiles"]:
            raise CatalogBuildError(
                f"{path}.benchmark_profile is not declared by benchmark {benchmark}"
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
