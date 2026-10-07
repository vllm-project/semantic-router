"""Known-issue waivers for live recipe conformance on CPU.

One exception to one outcome: the routing preview's own deadline on a named
long-context probe, while Vela 2.0 0.3B reads the whole input on a CPU runner
(#4706). Thresholds, deadlines, pass rates and probes stay unchanged. A
misroute, any other error, and a deadline on an unnamed probe still fail.

Remove this module when #4706 bounds that read: the named probes must then
pass without it.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any

from recipe_conformance_sources import discover_recipe_sources

ISSUE = 4706
KIND = "cpu-preview-deadline"
REASON = (
    "On a 4-vCPU CPU runner, Vela 2.0 0.3B reads every token of a long-context "
    "input, 8,192-token window by window, so these probes can outlast the routing "
    "preview's 120 s deadline."
)
REMOVAL = (
    "Remove this waiver when #4706 bounds Vela 2.0's CPU read; the named probes "
    "must then pass without it."
)
# A response that reaches the deadline can still return 200 with the signals
# the deadline cut short; it arrives within this much of the deadline.
DEADLINE_SLACK_SECONDS = 1.0
DEFAULT_PREVIEW_DEADLINE_SECONDS = 120.0

# Every long-context probe a 4-vCPU CI runner answered after 90 s or not at all.
WAIVED_PROBES: dict[str, dict[str, tuple[str, ...]]] = {
    "standalone": {
        "balance": ("casual_chat:long_unclassified_fallback",),
    },
    "built-in-latest": {
        "mom-v1": (
            "accuracy_context_partition:accuracy_long_context_or_direct_reference__long_workflow_collision",
            "accuracy_reasoning:accuracy_image_tool_result_synthesis__image_tool_result_beyond_240k",
            "accuracy_reasoning:accuracy_image_tool_result_synthesis__image_tool_result_from_120k_to_240k",
            "accuracy_reasoning:accuracy_text_tool_result_synthesis__text_tool_result_beyond_240k",
            "accuracy_reasoning:accuracy_text_tool_result_synthesis__text_tool_result_from_120k_to_240k",
            "accuracy_simple:accuracy_image_from_120k_to_240k__image_context_boundary_en",
            "accuracy_simple:accuracy_image_from_120k_to_240k__image_context_tools_collision_zh",
            "accuracy_simple:accuracy_long_context_or_direct_reference__long_context_direct",
            "accuracy_simple:accuracy_long_context_or_direct_reference__long_context_paraphrase",
            "accuracy_simple:accuracy_long_context_or_direct_reference__long_context_tools_collision",
            "accuracy_simple:accuracy_over_240k_image_guard__image_at_first_over_240k_token",
            "accuracy_simple:accuracy_over_240k_text_guard__text_at_first_over_240k_band",
            "accuracy_work_decomposition:accuracy_over_240k_text_guard__boundary_orchestration_tools_collision",
            "balance_medium:balance_over_240k_image_guard__image_at_first_over_240k_token",
            "balance_medium:balance_over_240k_text_guard__boundary_orchestration_tools_collision",
            "balance_medium:balance_text_from_30k_to_60k__direct_summary_in_30k_60k_band",
            "balance_medium:balance_text_from_60k_to_120k__direct_summary_in_60k_120k_band",
            "balance_medium:balance_text_from_60k_to_120k__direct_summary_in_60k_120k_band_zh",
            "balance_reasoning:balance_text_from_30k_to_60k__root_cause_in_30k_60k_band_zh",
            "balance_reasoning:balance_text_from_60k_to_120k__deliberate_tools_in_60k_120k_band_collision",
            "balance_simple:balance_image__image_in_120k_240k_band",
            "balance_simple:balance_image__image_in_30k_60k_band",
            "balance_simple:balance_image__image_in_60k_120k_band",
            "balance_simple:balance_image_tools__image_tools_in_120k_240k_band",
            "balance_simple:balance_image_tools__image_tools_in_30k_60k_band",
            "balance_simple:balance_image_tools__image_tools_in_60k_120k_band",
            "balance_simple:balance_over_240k_text_guard__text_at_first_over_240k_band",
            "balance_simple:balance_text_from_120k_to_240k__text_in_120k_240k_band",
            "balance_simple:balance_text_from_120k_to_240k__tools_in_120k_240k_band",
            "balance_simple:balance_text_from_30k_to_60k__tools_in_30k_60k_band_collision",
        ),
    },
}


def waiver_policy() -> dict[str, Any]:
    """The waiver as a CI plan records it: one case ID per named probe request."""
    return {
        "issue": ISSUE,
        "kind": KIND,
        "reason": REASON,
        "removal": REMOVAL,
        "cases": sorted(
            f"{source}:{recipe}:request:{probe_id}"
            for source, recipes in WAIVED_PROBES.items()
            for recipe, probe_ids in recipes.items()
            for probe_id in probe_ids
        ),
    }


def recipe_source_name(recipes_root: Path, default_root: Path) -> str | None:
    """The conformance source whose recipes live in ``recipes_root``, if any."""
    resolved = recipes_root.resolve()
    for source in discover_recipe_sources(default_root):
        if source.recipes_root.resolve() == resolved:
            return source.name
    return None


def deadline_outcome(result: dict[str, Any], deadline_seconds: float) -> str | None:
    """Which preview-deadline outcome a failed result shows, if it shows one."""
    response = result.get("raw_response")
    error = response.get("error") if isinstance(response, dict) else None
    if (
        result.get("http_status") == HTTPStatus.GATEWAY_TIMEOUT
        and isinstance(error, dict)
        and error.get("code") == "REQUEST_TIMEOUT"
    ):
        return "request_timeout"
    signal_errors = result.get("signal_errors") or {}
    if (
        result.get("http_status") == HTTPStatus.OK
        and signal_errors
        and all(
            str(code).endswith("_evaluation_failed") for code in signal_errors.values()
        )
        and float(result.get("latency_ms") or 0.0)
        >= (deadline_seconds - DEADLINE_SLACK_SECONDS) * 1000
    ):
        return "signals_cut_at_deadline"
    return None


def known_issue_waiver(
    source: str | None,
    recipe: str,
    result: dict[str, Any],
    deadline_seconds: float = DEFAULT_PREVIEW_DEADLINE_SECONDS,
) -> dict[str, Any] | None:
    """The waiver for one failed result, or None when it must fail as usual."""
    if result.get("matched"):
        return None
    if result.get("id") not in WAIVED_PROBES.get(source or "", {}).get(recipe, ()):
        return None
    outcome = deadline_outcome(result, deadline_seconds)
    if outcome is None:
        return None
    return {"issue": ISSUE, "kind": KIND, "outcome": outcome}


def preview_deadline_seconds(runtime_config: dict[str, Any] | None) -> float:
    """The routing preview deadline a runtime config sets (the Router's default otherwise)."""
    preview = (
        (((runtime_config or {}).get("global") or {}).get("services") or {}).get("api")
        or {}
    ).get("routing_preview") or {}
    value = preview.get("request_timeout_seconds")
    return float(value) if value else DEFAULT_PREVIEW_DEADLINE_SECONDS


def describe_waivers(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The waiver section of an evaluation: the policy and every waived result."""
    waived = [
        {
            "id": result["id"],
            "outcome": result["known_issue_waiver"]["outcome"],
            "latency_ms": result.get("latency_ms"),
        }
        for result in results
        if result.get("known_issue_waiver")
    ]
    if not waived:
        return None
    return {
        **{key: value for key, value in waiver_policy().items() if key != "cases"},
        "waived": waived,
    }
