from __future__ import annotations

from typing import Any

VSR_CANONICAL_PATCH_YAML = "vsr_canonical_patch.yaml"
VSR_CANONICAL_PATCH_JSON = "vsr_canonical_patch_recommendation.json"

ACCURACY_GAIN_THRESHOLD = 5.0
HIGH_OVERHEAD_THRESHOLD = 50.0
MODERATE_OVERHEAD_THRESHOLD = 20.0
GOOD_OVERHEAD_THRESHOLD = 30.0

REASONING_FAMILY_PATCHES: dict[str, dict[str, str]] = {
    "qwen3": {"type": "chat_template_kwargs", "parameter": "enable_thinking"},
    "deepseek": {"type": "chat_template_kwargs", "parameter": "thinking"},
    "gpt-oss": {"type": "reasoning_effort", "parameter": "reasoning_effort"},
}


def build_vsr_canonical_patch(
    model_name: str, reasoning_family: str | None
) -> tuple[dict[str, Any], str | None]:
    """Build a canonical v0.3 patch for reasoning-family wiring."""
    card_patch: dict[str, Any] = {"name": model_name}
    provider_model_patch: dict[str, Any] = {"name": model_name}
    patch: dict[str, Any] = {
        "providers": {"models": [provider_model_patch]},
        "routing": {"modelCards": [card_patch]},
    }

    if reasoning_family in REASONING_FAMILY_PATCHES:
        patch["providers"]["defaults"] = {
            "reasoning_families": {
                reasoning_family: REASONING_FAMILY_PATCHES[reasoning_family]
            }
        }
        provider_model_patch["reasoning_family"] = reasoning_family
        return patch, None

    manual_follow_up = (
        "Set providers.models[].reasoning_family and add a matching "
        "providers.defaults.reasoning_families entry before merging this patch. "
        "Supported families: qwen3, deepseek, gpt-oss."
    )
    return patch, manual_follow_up


def generate_vsr_canonical_patch(
    comparisons: list[Any],
    model_name: str,
    reasoning_family: str | None = None,
) -> dict[str, Any]:
    """Generate a canonical vSR config patch based on evaluation results."""
    total_std_accuracy = _average_metric(comparisons, "standard_mode", "accuracy")
    total_reas_accuracy = _average_metric(comparisons, "reasoning_mode", "accuracy")
    avg_token_ratio_std = _average_metric(
        comparisons, "standard_mode", "token_usage_ratio"
    )
    avg_token_ratio_reas = _average_metric(
        comparisons, "reasoning_mode", "token_usage_ratio"
    )
    avg_time_per_token_std = _average_metric(
        comparisons, "standard_mode", "time_per_output_token"
    )
    avg_time_per_token_reas = _average_metric(
        comparisons, "reasoning_mode", "time_per_output_token"
    )

    resolved_family = reasoning_family or "auto"
    suggested_patch, manual_follow_up = build_vsr_canonical_patch(
        model_name, resolved_family
    )

    accuracy_improvement = _relative_delta(total_reas_accuracy, total_std_accuracy)
    token_overhead = _relative_delta(avg_token_ratio_reas, avg_token_ratio_std)
    latency_overhead = _relative_delta(avg_time_per_token_reas, avg_time_per_token_std)

    recommendation = {
        "model_name": model_name,
        "reasoning_family": resolved_family,
        "performance_analysis": {
            "standard_mode": {
                "avg_accuracy": round(total_std_accuracy, 4),
                "avg_token_usage_ratio": round(avg_token_ratio_std, 4),
                "avg_time_per_output_token_ms": round(avg_time_per_token_std, 2),
            },
            "reasoning_mode": {
                "avg_accuracy": round(total_reas_accuracy, 4),
                "avg_token_usage_ratio": round(avg_token_ratio_reas, 4),
                "avg_time_per_output_token_ms": round(avg_time_per_token_reas, 2),
            },
            "improvements": {
                "accuracy_change_percent": round(accuracy_improvement, 2),
                "token_overhead_percent": round(token_overhead, 2),
                "latency_overhead_percent": round(latency_overhead, 2),
            },
        },
        "recommendation": generate_recommendation_text(
            accuracy_improvement, token_overhead, latency_overhead
        ),
        "merge_instructions": (
            "Merge the generated patch into config/config.yaml. "
            "It updates providers.defaults.reasoning_families and "
            "providers.models plus routing.modelCards for the evaluated model."
        ),
        "suggested_vsr_patch": suggested_patch,
    }
    if manual_follow_up:
        recommendation["manual_follow_up"] = manual_follow_up
    return recommendation


def _average_metric(comparisons: list[Any], mode_attr: str, metric_attr: str) -> float:
    values = [getattr(getattr(comp, mode_attr), metric_attr) for comp in comparisons]
    return sum(values) / len(values)


def _relative_delta(new_value: float, old_value: float) -> float:
    if old_value <= 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def generate_recommendation_text(
    accuracy_imp: float, token_overhead: float, latency_overhead: float
) -> str:
    """Generate human-readable recommendation based on metrics."""
    recommendations: list[str] = []

    if accuracy_imp > ACCURACY_GAIN_THRESHOLD:
        recommendations.append(
            f"✅ Reasoning mode shows {accuracy_imp:.1f}% accuracy improvement. "
            "Recommended for accuracy-critical applications."
        )
    elif accuracy_imp > 0:
        recommendations.append(
            f"⚖️ Reasoning mode shows modest {accuracy_imp:.1f}% accuracy improvement."
        )
    else:
        recommendations.append(
            f"⚠️ Reasoning mode shows {abs(accuracy_imp):.1f}% accuracy degradation. "
            "Standard mode may be preferable."
        )

    recommendations.extend(_overhead_text("token usage", token_overhead))
    recommendations.extend(_overhead_text("latency per token", latency_overhead))

    if (
        accuracy_imp > ACCURACY_GAIN_THRESHOLD
        and token_overhead < GOOD_OVERHEAD_THRESHOLD
        and latency_overhead < GOOD_OVERHEAD_THRESHOLD
    ):
        recommendations.append(
            "🎯 Overall: Reasoning mode offers good accuracy improvement with acceptable overhead. Recommended for production use."
        )
    elif accuracy_imp < 0:
        recommendations.append(
            "🎯 Overall: Standard mode is recommended based on these results."
        )
    else:
        recommendations.append(
            "🎯 Overall: Consider enabling reasoning mode selectively for complex routes or high-priority traffic."
        )
    return "\n".join(recommendations)


def _overhead_text(metric_name: str, overhead: float) -> list[str]:
    if overhead > HIGH_OVERHEAD_THRESHOLD:
        return [
            f"💰 Reasoning mode has {overhead:.1f}% higher {metric_name}. "
            "Consider cost and latency implications."
        ]
    if overhead > MODERATE_OVERHEAD_THRESHOLD:
        return [f"📊 Reasoning mode has {overhead:.1f}% moderate {metric_name}."]
    return []
