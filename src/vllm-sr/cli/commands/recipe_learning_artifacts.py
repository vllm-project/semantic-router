"""Artifact builders for offline Router Learning recipe analysis."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from cli.commands.recipe_learning_metrics import numeric

_RECIPE_DECISION_PATH_MIN_PARTS = 3
_PRIORITY_STEP = 10
_STABILITY_WEIGHT_INCREASE = 1.25
_STABILITY_WEIGHT_DECREASE = 0.75
_STABILITY_MIN_INCREASED_WEIGHT = 1.25
_COUNTERFACTUAL_SMALL_GAIN = 0.05
_COUNTERFACTUAL_SWITCH_REDUCTION = 0.85
_COUNTERFACTUAL_COST_SAVINGS_GAIN = 1.02
_COUNTERFACTUAL_RATE_REDUCTION = 0.8
_COUNTERFACTUAL_RATE_INCREASE = 1.1


def empty_recipe_patch(mode: str) -> dict[str, Any]:
    return {
        "format": "router_learning.suggested_config_patch",
        "mode": mode,
        "suggestions": [],
    }


def build_recipe_patch_suggestions(findings: list[dict[str, Any]]) -> dict[str, Any]:
    suggestions = []
    for item in findings:
        decision = item.get("decision", "")
        finding_type = item.get("type")
        if finding_type == "wrong_decision":
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/priority",
                    "decrease",
                    "Wrong-route eval evidence suggests this decision may outrank a better match.",
                    finding=item,
                )
            )
        elif finding_type == "wrong_model_selection":
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/adaptations/adaptation/candidate_set",
                    "set",
                    "Model-fit eval evidence suggests this decision should let adaptation compare tier candidates.",
                    value="tier",
                    finding=item,
                )
            )
        elif finding_type == "excessive_switching":
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/adaptations/protection/stability_weight",
                    "increase",
                    "Frequent switches suggest stronger protection for this decision.",
                    finding=item,
                )
            )
        elif finding_type == "excessive_holds":
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/adaptations/protection/stability_weight",
                    "decrease",
                    "Protection is holding while model-fit evidence suggests the current model is weak.",
                    finding=item,
                )
            )
        elif finding_type == "missing_protection":
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/adaptations/protection/mode",
                    "set",
                    "Protection diagnostics are missing for records where adaptation is active.",
                    value="apply",
                    finding=item,
                )
            )
        elif finding_type == "overly_broad_candidate_set":
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/adaptations/adaptation/candidate_set",
                    "set",
                    "Broad candidate sets should be narrowed when they correlate with switching or overuse.",
                    value="decision",
                    finding=item,
                )
            )
        elif finding_type in {"latency_violation", "cost_violation"}:
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/adaptations/adaptation/candidate_set",
                    "set",
                    "Budget violations suggest using the narrowest candidate set until modelRefs are reviewed.",
                    value="decision",
                    finding=item,
                )
            )
        elif finding_type in {"underpowered_model", "model_overuse"}:
            suggestions.append(
                patch_suggestion(
                    f"/routing/decisions/{decision}/adaptations/adaptation/candidate_set",
                    "set",
                    item.get("recommendation", ""),
                    value="tier",
                    finding=item,
                )
            )
    return {
        "format": "router_learning.suggested_config_patch",
        "mode": "patch_generating",
        "suggestions": suggestions,
    }


def build_candidate_recipes(
    recipe: dict[str, Any] | None,
    recipe_patch: dict[str, Any],
) -> list[dict[str, Any]]:
    suggestions = [
        item for item in recipe_patch.get("suggestions", []) if isinstance(item, dict)
    ]
    if not suggestions:
        return []

    candidates = []
    for index, suggestion in enumerate(suggestions, start=1):
        candidate_id = f"candidate_{index:02d}"
        candidate: dict[str, Any] = {
            "id": candidate_id,
            "description": suggestion.get("reason", ""),
            "patch": {
                "format": recipe_patch.get("format"),
                "suggestions": [suggestion],
            },
        }
        if recipe is not None:
            candidate["recipe"] = apply_recipe_suggestion(recipe, suggestion)
        else:
            candidate["recipe"] = None
            candidate["note"] = (
                "Pass --recipe-file to materialize this patch as a complete candidate recipe."
            )
        candidates.append(candidate)
    return candidates


def apply_recipe_suggestion(
    recipe: dict[str, Any],
    suggestion: dict[str, Any],
) -> dict[str, Any]:
    candidate = deepcopy(recipe)
    path = parse_decision_patch_path(str(suggestion.get("path") or ""))
    if path is None:
        return candidate
    decision = find_recipe_decision(candidate, path.decision)
    if decision is None:
        return candidate
    adaptations = decision.setdefault("adaptations", {})
    apply_decision_patch(decision, adaptations, path, suggestion)
    return candidate


@dataclass(frozen=True)
class DecisionPatchPath:
    decision: str
    parts: tuple[str, ...]

    def endswith(self, *suffix: str) -> bool:
        return self.parts[-len(suffix) :] == suffix


def parse_decision_patch_path(path: str) -> DecisionPatchPath | None:
    parts = tuple(part for part in path.split("/") if part)
    if len(parts) < _RECIPE_DECISION_PATH_MIN_PARTS or parts[:2] != (
        "routing",
        "decisions",
    ):
        return None
    return DecisionPatchPath(decision=parts[2], parts=parts)


def apply_decision_patch(
    decision: dict[str, Any],
    adaptations: dict[str, Any],
    path: DecisionPatchPath,
    suggestion: dict[str, Any],
) -> None:
    action = str(suggestion.get("action") or "")
    if path.endswith("priority"):
        apply_priority_patch(decision, action)
    elif path.endswith("protection", "stability_weight"):
        apply_protection_stability_weight_patch(adaptations, action)
    elif path.endswith("adaptation", "candidate_set"):
        apply_adaptation_candidate_set_patch(adaptations, action, suggestion)
    elif path.endswith("protection", "mode"):
        apply_protection_mode_patch(adaptations, action, suggestion)


def apply_priority_patch(decision: dict[str, Any], action: str) -> None:
    current_priority = int(numeric(decision.get("priority")))
    if action == "increase":
        decision["priority"] = current_priority + _PRIORITY_STEP
    elif action == "decrease":
        decision["priority"] = max(0, current_priority - _PRIORITY_STEP)


def apply_protection_stability_weight_patch(
    adaptations: dict[str, Any],
    action: str,
) -> None:
    protection = adaptations.setdefault("protection", {})
    current = numeric(protection.get("stability_weight")) or 1.0
    if action == "increase":
        protection["stability_weight"] = round(
            max(
                _STABILITY_MIN_INCREASED_WEIGHT,
                current * _STABILITY_WEIGHT_INCREASE,
            ),
            3,
        )
    elif action == "decrease":
        protection["stability_weight"] = round(
            max(0.0, current * _STABILITY_WEIGHT_DECREASE),
            3,
        )


def apply_adaptation_candidate_set_patch(
    adaptations: dict[str, Any],
    action: str,
    suggestion: dict[str, Any],
) -> None:
    value = str(suggestion.get("value") or "tier")
    if action == "set" and value in {"decision", "tier", "global"}:
        adaptation = adaptations.setdefault("adaptation", {})
        adaptation["candidate_set"] = value


def apply_protection_mode_patch(
    adaptations: dict[str, Any],
    action: str,
    suggestion: dict[str, Any],
) -> None:
    value = str(suggestion.get("value") or "apply")
    if action == "set" and value in {"apply", "observe", "bypass"}:
        protection = adaptations.setdefault("protection", {})
        protection["mode"] = value


def find_recipe_decision(
    recipe: dict[str, Any], decision_name: str
) -> dict[str, Any] | None:
    routing = recipe.get("routing")
    if not isinstance(routing, dict):
        return None
    decisions = routing.get("decisions")
    if not isinstance(decisions, list):
        return None
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        if str(decision.get("name") or decision.get("id") or "") == decision_name:
            return decision
    return None


def build_recipe_learning_experiments(
    metrics_payload: dict[str, Any],
    candidate_recipes: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = deepcopy(metrics_payload)
    return {
        "method": "replay_counterfactual",
        "baseline": {
            "id": "baseline",
            "metrics": baseline,
        },
        "candidates": [
            run_candidate_recipe_experiment(metrics_payload, candidate)
            for candidate in candidate_recipes
        ],
    }


def run_candidate_recipe_experiment(
    metrics_payload: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline = deepcopy(metrics_payload)
    suggestions = candidate.get("patch", {}).get("suggestions", [])
    candidate_metrics = deepcopy(metrics_payload)
    applied_suggestions = []
    limitations = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue
        decision = decision_from_patch_path(str(suggestion.get("path") or ""))
        if not decision:
            limitations.append(
                f"Suggestion path {suggestion.get('path')!r} is not replay-evaluable."
            )
            continue
        metrics = candidate_metrics.get("per_decision", {}).get(decision)
        if not isinstance(metrics, dict):
            limitations.append(
                f"No replay metrics exist for decision {decision!r}; suggestion kept as a patch only."
            )
            continue
        if apply_replay_counterfactual(metrics, suggestion):
            apply_tier_counterfactual(candidate_metrics, metrics, suggestion)
            applied_suggestions.append(suggestion)
        else:
            limitations.append(
                f"Suggestion action {suggestion.get('action')!r} at {suggestion.get('path')!r} "
                "does not have a deterministic replay counterfactual yet."
            )
    return {
        "id": candidate.get("id"),
        "recipe_digest": recipe_digest(candidate.get("recipe")),
        "applied_suggestions": applied_suggestions,
        "metrics": candidate_metrics,
        "deltas": metric_deltas(baseline, candidate_metrics),
        "limitations": limitations,
    }


def apply_replay_counterfactual(
    metrics: dict[str, Any],
    suggestion: dict[str, Any],
) -> bool:
    path = str(suggestion.get("path") or "")
    action = str(suggestion.get("action") or "")
    value = str(suggestion.get("value") or "")
    if path.endswith("/priority") and action == "decrease":
        route_correctness = metrics.get("route_correctness")
        if route_correctness is not None:
            metrics["route_correctness"] = round(
                min(1.0, numeric(route_correctness) + _COUNTERFACTUAL_SMALL_GAIN),
                4,
            )
        return True
    if path.endswith("/adaptations/protection/mode") and value == "apply":
        metrics["missing_protection_rate"] = 0.0
        return True
    if path.endswith("/adaptations/adaptation/candidate_set") and value == "decision":
        metrics["broad_candidate_rate"] = 0.0
        metrics["switch_rate"] = round(
            numeric(metrics.get("switch_rate")) * _COUNTERFACTUAL_SWITCH_REDUCTION,
            4,
        )
        metrics["cost_savings"] = round(
            numeric(metrics.get("cost_savings")) * _COUNTERFACTUAL_COST_SAVINGS_GAIN,
            6,
        )
        return True
    if path.endswith("/adaptations/adaptation/candidate_set") and value == "tier":
        model_fit = metrics.get("model_fit")
        if model_fit is not None:
            metrics["model_fit"] = round(
                min(1.0, numeric(model_fit) + _COUNTERFACTUAL_SMALL_GAIN),
                4,
            )
        return True
    if not path.endswith("/adaptations/protection/stability_weight"):
        return False
    if action == "increase":
        metrics["switch_rate"] = round(
            numeric(metrics.get("switch_rate")) * _COUNTERFACTUAL_RATE_REDUCTION,
            4,
        )
        metrics["hold_rate"] = round(
            min(1.0, numeric(metrics.get("hold_rate")) * _COUNTERFACTUAL_RATE_INCREASE),
            4,
        )
        return True
    if action == "decrease":
        metrics["hold_rate"] = round(
            numeric(metrics.get("hold_rate")) * _COUNTERFACTUAL_RATE_REDUCTION,
            4,
        )
        metrics["switch_rate"] = round(
            min(
                1.0,
                numeric(metrics.get("switch_rate")) * _COUNTERFACTUAL_RATE_INCREASE,
            ),
            4,
        )
        return True
    return False


def apply_tier_counterfactual(
    candidate_metrics: dict[str, Any],
    decision_metrics: dict[str, Any],
    suggestion: dict[str, Any],
) -> None:
    tier_key = dominant_tier_key(decision_metrics)
    if not tier_key:
        return
    per_tier = candidate_metrics.get("per_tier")
    if not isinstance(per_tier, dict):
        return
    tier_metrics = per_tier.get(tier_key)
    if not isinstance(tier_metrics, dict):
        return
    apply_replay_counterfactual(tier_metrics, suggestion)


def dominant_tier_key(metrics: dict[str, Any]) -> str:
    tiers = metrics.get("decision_tiers")
    if not isinstance(tiers, dict):
        return ""
    ranked = sorted(
        (
            (str(tier), numeric(count))
            for tier, count in tiers.items()
            if str(tier) != "unknown" and numeric(count) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0][0] if ranked else ""


def metric_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "overall": metric_delta_block(
            baseline.get("overall", {}),
            candidate.get("overall", {}),
        ),
        "per_decision": {
            decision: metric_delta_block(
                baseline.get("per_decision", {}).get(decision, {}),
                metrics,
            )
            for decision, metrics in sorted(candidate.get("per_decision", {}).items())
            if isinstance(metrics, dict)
        },
        "per_tier": {
            tier: metric_delta_block(
                baseline.get("per_tier", {}).get(tier, {}),
                metrics,
            )
            for tier, metrics in sorted(candidate.get("per_tier", {}).items())
            if isinstance(metrics, dict)
        },
    }


def metric_delta_block(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    keys = (
        "route_correctness",
        "model_fit",
        "switch_rate",
        "hold_rate",
        "sampling_rate",
        "rescue_rate",
        "missing_protection_rate",
        "broad_candidate_rate",
        "cache_preservation",
        "avg_latency_ms",
        "actual_cost",
        "baseline_cost",
        "cost_savings",
    )
    deltas: dict[str, float] = {}
    for key in keys:
        base_value = baseline.get(key)
        candidate_value = candidate.get(key)
        if base_value is None or candidate_value is None:
            continue
        deltas[key] = round(numeric(candidate_value) - numeric(base_value), 6)
    return deltas


def recipe_digest(recipe: Any) -> str:
    if recipe is None:
        return ""
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def decision_from_patch_path(path: str) -> str:
    parsed = parse_decision_patch_path(path)
    return parsed.decision if parsed is not None else ""


def build_experience_seed_pack(
    experience_counts: dict[tuple[str, int, str], dict[str, int]],
) -> dict[str, Any]:
    records = []
    for (decision, tier, model), counts in sorted(experience_counts.items()):
        good_fit = counts["good_fit"]
        underpowered = counts["underpowered"]
        fit_total = good_fit + underpowered
        if fit_total == 0:
            quality_seed = 0.5
            seed_weight = counts["overprovisioned"] + counts["failed"]
        else:
            quality_seed = good_fit / fit_total
            seed_weight = fit_total
        records.append(
            {
                "decision_id": decision,
                "decision_tier": tier,
                "model": model,
                "quality_seed": round(quality_seed, 4),
                "seed_weight": seed_weight,
                "source_metric": "model_outcomes",
                "support": {
                    "model_outcomes": sum(counts.values()),
                },
                "good_fit_count": good_fit,
                "underpowered_count": underpowered,
                "overprovisioned_count": counts["overprovisioned"],
                "failed_count": counts["failed"],
            }
        )
    return {
        "version": 1,
        "records": records,
    }


def patch_suggestion(
    path: str,
    action: str,
    reason: str,
    value: Any = None,
    finding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suggestion = {
        "path": path,
        "action": action,
        "reason": reason,
    }
    if value is not None:
        suggestion["value"] = value
    if finding:
        suggestion["finding_id"] = finding.get("id")
        suggestion["finding_type"] = finding.get("type")
        suggestion["decision"] = finding.get("decision")
    return suggestion
