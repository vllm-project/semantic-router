"""Multidimensional coverage accounting and ratchets for recipe probes."""

from __future__ import annotations

from collections import Counter
from typing import Any

from router_calibration_probe import Probe

SIGNAL_RESPONSE_TYPES = {
    "keywords": "keywords",
    "embeddings": "embeddings",
    "domains": "domains",
    "fact_check": "fact_check",
    "user_feedbacks": "user_feedback",
    "reasks": "reask",
    "preferences": "preferences",
    "language": "language",
    "context": "context",
    "structure": "structure",
    "complexity": "complexity",
    "modality": "modality",
    "role_bindings": "authz",
    "jailbreak": "jailbreak",
    "pii": "pii",
    "kb": "kb",
    "conversation": "conversation",
    "events": "event",
    "metadata": "metadata",
    "classifiers": "classifier",
    "input_modality": "input_modality",
}
TAG_CATEGORIES = (
    "language",
    "negative",
    "collision",
    "fallback",
    "stress",
    "tools",
    "multi_turn",
)
PERCENT_POLICY_FIELDS = {
    "signals": "min_signal_assertion_percent",
    "projections": "min_projection_assertion_percent",
    "algorithms": "min_algorithm_assertion_percent",
    "plugins": "min_plugin_assertion_percent",
}


def configured_signal_names(profile: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for collection_name, raw_rules in _mapping(profile.get("signals")).items():
        names = {
            str(_mapping(raw_rule).get("name") or "").strip()
            for raw_rule in _sequence(raw_rules)
        }
        signal_type = SIGNAL_RESPONSE_TYPES.get(
            str(collection_name), str(collection_name)
        )
        result[signal_type] = {name for name in names if name}
    projection_names: set[str] = set()
    projections = _mapping(profile.get("projections"))
    for raw_mapping in _sequence(projections.get("mappings")):
        for raw_output in _sequence(_mapping(raw_mapping).get("outputs")):
            name = str(_mapping(raw_output).get("name") or "").strip()
            if name:
                projection_names.add(name)
    if projection_names:
        result["projection"] = projection_names
    return result


def collect_coverage(
    *,
    profiles: dict[str, dict[str, Any]],
    decisions: dict[tuple[str, str], dict[str, Any]],
    entrypoints: dict[str, str],
    probes: list[Probe],
    policy: dict[str, Any],
) -> dict[str, Any]:
    configured_signals = _configured_signals(profiles)
    configured_decisions = {
        _decision_key(recipe, decision) for recipe, decision in decisions
    }
    configured_algorithms = _configured_algorithms(decisions)
    configured_plugins = _configured_plugins(decisions)
    configured_fallbacks = {
        _decision_key(recipe, name)
        for (recipe, name), decision in decisions.items()
        if not _sequence(_mapping(decision.get("rules")).get("conditions"))
    }

    asserted_signals: set[str] = set()
    asserted_decisions: set[str] = set()
    asserted_algorithms: set[str] = set()
    asserted_plugins: set[str] = set()
    asserted_entrypoints: set[str] = set()
    request_shapes: set[str] = set()
    tag_counts: Counter[str] = Counter()
    for probe in probes:
        recipe = _probe_recipe(probe, entrypoints)
        decision_key = _decision_key(recipe, probe.expected_decision)
        asserted_decisions.add(decision_key)
        asserted_signals.update(
            _signal_key(signal_type, name)
            for signal_type, name in (
                *probe.expected_signals,
                *probe.forbidden_signals,
            )
        )
        if probe.expected_algorithm:
            asserted_algorithms.add(
                _decision_surface_key(
                    recipe, probe.expected_decision, probe.expected_algorithm
                )
            )
        asserted_plugins.update(
            _decision_surface_key(recipe, probe.expected_decision, plugin)
            for plugin in probe.expected_plugins
        )
        if probe.model:
            asserted_entrypoints.add(probe.model)
        request_shapes.add("messages" if probe.messages else "text")
        if probe.tools:
            request_shapes.add("tools")
        categories = tag_categories(probe.tags)
        if probe.messages:
            categories.add("multi_turn")
        if probe.tools:
            categories.add("tools")
        tag_counts.update(categories)

    signal_record = _coverage_record(configured_signals, asserted_signals)
    projection_configured = {
        key for key in configured_signals if key.startswith("projection:")
    }
    projection_asserted = {
        key for key in asserted_signals if key.startswith("projection:")
    }
    records = {
        "signals": {
            **signal_record,
            "by_family": _signal_family_records(configured_signals, asserted_signals),
        },
        "projections": _coverage_record(projection_configured, projection_asserted),
        "decisions": _coverage_record(configured_decisions, asserted_decisions),
        "entrypoints": _coverage_record(set(entrypoints), asserted_entrypoints),
        "fallbacks": _coverage_record(configured_fallbacks, asserted_decisions),
        "algorithms": _coverage_record(configured_algorithms, asserted_algorithms),
        "plugins": _coverage_record(configured_plugins, asserted_plugins),
        "request_shapes": sorted(request_shapes),
        "tag_counts": {
            category: tag_counts.get(category, 0) for category in TAG_CATEGORIES
        },
        "policy": policy,
    }
    errors = static_policy_errors(records, policy)
    records["errors"] = errors
    records["passed"] = not errors
    return records


def static_policy_errors(coverage: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for dimension, policy_field in PERCENT_POLICY_FIELDS.items():
        minimum = float(policy.get(policy_field, 0))
        actual = float(coverage[dimension]["percent"])
        if actual < minimum:
            errors.append(
                f"{dimension} assertion coverage {actual:.1f}% is below {minimum:.1f}%"
            )
    required_shapes = set(policy.get("required_request_shapes", []))
    missing_shapes = sorted(required_shapes - set(coverage["request_shapes"]))
    if missing_shapes:
        errors.append(
            "required request shapes are missing: " + ", ".join(missing_shapes)
        )
    for category, minimum in policy.get("min_tag_counts", {}).items():
        actual = int(coverage["tag_counts"].get(category, 0))
        if actual < int(minimum):
            errors.append(
                f"tag category {category!r} has {actual} probes, requires {minimum}"
            )
    return errors


def summarize_catalog_coverage(
    coverages: list[dict[str, Any]],
) -> dict[str, Any]:
    dimensions = (
        "signals",
        "projections",
        "decisions",
        "entrypoints",
        "fallbacks",
        "algorithms",
        "plugins",
    )
    result: dict[str, Any] = {}
    for dimension in dimensions:
        covered = sum(int(coverage[dimension]["covered"]) for coverage in coverages)
        total = sum(int(coverage[dimension]["total"]) for coverage in coverages)
        result[dimension] = {
            "covered": covered,
            "total": total,
            "percent": (round((covered / total) * 100, 1) if total else 100.0),
        }
    result["passed"] = all(coverage.get("passed") for coverage in coverages)
    return result


def evaluate_live_tag_policy(
    evaluation: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    category_results: dict[str, list[bool]] = {
        category: [] for category in TAG_CATEGORIES
    }
    for result in evaluation.get("results", []):
        categories = tag_categories(tuple(result.get("tags", [])))
        if result.get("messages"):
            categories.add("multi_turn")
        if result.get("tools"):
            categories.add("tools")
        for category in categories:
            category_results[category].append(bool(result.get("matched")))
    summaries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for category, minimum in policy.get("min_tag_pass_rate", {}).items():
        results = category_results.get(category, [])
        passed = sum(results)
        rate = round((passed / len(results)) * 100, 1) if results else 0.0
        summaries[category] = {
            "matched": passed,
            "total": len(results),
            "pass_rate": rate,
            "minimum": float(minimum),
        }
        if rate < float(minimum):
            errors.append(
                f"tag category {category!r} pass rate {rate:.1f}% is below "
                f"{float(minimum):.1f}%"
            )
    return {"passed": not errors, "errors": errors, "categories": summaries}


def tag_categories(tags: tuple[str, ...]) -> set[str]:
    normalized = {tag.strip().lower() for tag in tags}
    categories: set[str] = set()
    if any(tag.startswith("language:") for tag in normalized) or normalized & {
        "de",
        "en",
        "es",
        "fr",
        "ja",
        "zh",
    }:
        categories.add("language")
    if normalized & {"negative", "class:negative", "fallback"}:
        categories.add("negative")
    if "collision" in normalized or any(
        tag.endswith(":collision") for tag in normalized
    ):
        categories.add("collision")
    if "fallback" in normalized:
        categories.add("fallback")
    if any(tag.startswith("stress:") for tag in normalized):
        categories.add("stress")
    if normalized & {"tools", "shape:tools"}:
        categories.add("tools")
    if normalized & {"multi-turn", "multiturn", "shape:messages"}:
        categories.add("multi_turn")
    return categories


def _configured_signals(
    profiles: dict[str, dict[str, Any]],
) -> set[str]:
    return {
        _signal_key(signal_type, name)
        for profile in profiles.values()
        for signal_type, names in configured_signal_names(profile).items()
        for name in names
    }


def _configured_algorithms(
    decisions: dict[tuple[str, str], dict[str, Any]],
) -> set[str]:
    return {
        _decision_surface_key(
            recipe,
            name,
            str(_mapping(decision.get("algorithm")).get("type") or "static"),
        )
        for (recipe, name), decision in decisions.items()
    }


def _configured_plugins(
    decisions: dict[tuple[str, str], dict[str, Any]],
) -> set[str]:
    return {
        _decision_surface_key(
            recipe,
            name,
            str(_mapping(plugin).get("type") or "").strip(),
        )
        for (recipe, name), decision in decisions.items()
        for plugin in _sequence(decision.get("plugins"))
        if str(_mapping(plugin).get("type") or "").strip()
    }


def _signal_family_records(
    configured: set[str], asserted: set[str]
) -> dict[str, dict[str, Any]]:
    families = sorted(key.split(":", 1)[0] for key in configured)
    return {
        family: _coverage_record(
            {key for key in configured if key.startswith(f"{family}:")},
            {key for key in asserted if key.startswith(f"{family}:")},
        )
        for family in dict.fromkeys(families)
    }


def _coverage_record(configured: set[str], asserted: set[str]) -> dict[str, Any]:
    covered = configured & asserted
    total = len(configured)
    percent = round((len(covered) / total) * 100, 1) if total else 100.0
    return {
        "configured": sorted(configured),
        "asserted": sorted(covered),
        "uncovered": sorted(configured - covered),
        "covered": len(covered),
        "total": total,
        "percent": percent,
        "gap_reason": (
            None
            if len(covered) == total
            else "ratcheted baseline; add explicit probe evidence before "
            "raising the minimum"
        ),
    }


def _probe_recipe(probe: Probe, entrypoints: dict[str, str]) -> str:
    if probe.expected_recipe:
        return probe.expected_recipe
    if probe.model and probe.model in entrypoints:
        return entrypoints[probe.model]
    return "default"


def _decision_key(recipe: str, decision: str) -> str:
    return f"{recipe}:{decision}"


def _decision_surface_key(recipe: str, decision: str, surface: str) -> str:
    return f"{recipe}:{decision}:{surface}"


def _signal_key(signal_type: str, name: str) -> str:
    return f"{signal_type}:{name}"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
