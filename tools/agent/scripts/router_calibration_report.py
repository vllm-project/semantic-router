"""Reporting helpers for the router calibration loop."""

from __future__ import annotations

from typing import Any

from router_calibration_support import utc_now


def render_markdown_summary(
    probe_manifest: dict[str, Any],
    pre_eval: dict[str, Any] | None,
    post_eval: dict[str, Any] | None,
    validate_result: dict[str, Any] | None,
    deploy_result: dict[str, Any] | None,
) -> str:
    title = str(
        probe_manifest.get("name") or probe_manifest.get("profile") or "routing"
    )
    lines = _render_header(title, pre_eval, post_eval, deploy_result)
    lines.extend(_render_review_axes())
    if validate_result is not None:
        lines.extend(_render_validate_section(validate_result))
    after_eval = post_eval or pre_eval or {}
    decision_summaries = after_eval.get("decisions", [])
    after_results = after_eval.get("results", [])
    acceptance = after_eval.get("acceptance", {})
    lines.extend(_render_decision_section(decision_summaries, acceptance))
    lines.extend(_render_variant_section(after_results))
    lines.extend(_render_review_queue(decision_summaries, after_results))
    return "\n".join(lines).rstrip() + "\n"


def _render_header(
    title: str,
    pre_eval: dict[str, Any] | None,
    post_eval: dict[str, Any] | None,
    deploy_result: dict[str, Any] | None,
) -> list[str]:
    lines = [
        f"# Routing Calibration Summary: {title}",
        "",
        f"- Generated at: `{utc_now()}`",
    ]
    if pre_eval is not None:
        lines.extend(_render_eval_summary("Pre-deploy", pre_eval))
    if post_eval is not None:
        lines.extend(_render_eval_summary("Post-deploy", post_eval))
    if deploy_result is not None and isinstance(deploy_result, dict):
        version = deploy_result.get("version") or "unknown"
        lines.append(f"- Deploy version: `{version}`")
    return lines


def _render_eval_summary(label: str, evaluation: dict[str, Any]) -> list[str]:
    return [
        f"- {label} success: `{evaluation['matched']}/{evaluation['total']}` ({evaluation['success_rate']}%)",
        f"- {label} decision coverage: `{evaluation['matched_decisions']}/{evaluation['total_decisions']}` ({evaluation['decision_success_rate']}%)",
    ]


def _render_review_axes() -> list[str]:
    return [
        "",
        "## Review Axes",
        "",
        "0. `query_quality`: Is the probe semantically representative, or is it just a brittle trigger phrase?",
        "1. `routing_design`: Are the signal / projection / decision boundaries robust, or only sufficient for the current examples?",
        "2. `validator_quality`: Do local warnings reflect real ambiguity, or missing static semantics?",
        "",
    ]


def _render_validate_section(validate_result: dict[str, Any]) -> list[str]:
    return [
        "## Local Validate",
        "",
        f"- Valid: `{validate_result.get('valid')}`",
        f"- Return code: `{validate_result.get('returncode')}`",
        "",
    ]


def _render_decision_section(
    decision_summaries: list[dict[str, Any]], acceptance: dict[str, Any]
) -> list[str]:
    if not decision_summaries:
        return []
    lines = [
        "## Decision Robustness",
        "",
        f"- Minimum probe pass rate: `{acceptance.get('min_probe_pass_rate', 100.0)}%`",
        f"- Minimum decision pass rate: `{acceptance.get('min_decision_pass_rate', 100.0)}%`",
        "",
        "| Decision | Variants | Pass rate | Threshold | Result |",
        "|---|---|---|---|---|",
    ]
    for summary in decision_summaries:
        status = "pass" if summary["passed"] else "review"
        lines.append(
            f"| `{summary['decision_id']}` | `{summary['matched']}/{summary['total']}` | "
            f"`{summary['pass_rate']}%` | `{summary['required_pass_rate']}%` | `{status}` |"
        )
    lines.append("")
    return lines


def _render_variant_section(after_results: list[dict[str, Any]]) -> list[str]:
    if not after_results:
        return []
    lines = [
        "## Variant Outcomes",
        "",
        "| Variant | Expected | Actual | Tags | Result |",
        "|---|---|---|---|---|",
    ]
    for result in after_results:
        status = "pass" if result["matched"] else "review"
        actual = result["actual_decision"] or "(none)"
        tags = ",".join(result.get("tags") or []) or "-"
        lines.append(
            f"| `{result['id']}` | `{result['expected_decision']}` | `{actual}` | `{tags}` | `{status}` |"
        )
    lines.append("")
    return lines


def _render_review_queue(
    decision_summaries: list[dict[str, Any]], after_results: list[dict[str, Any]]
) -> list[str]:
    failing = [result for result in after_results if not result["matched"]]
    if not failing:
        return []
    lines = ["## Review Queue", ""]
    for summary in decision_summaries:
        decision_failures = [
            result
            for result in failing
            if result["decision_id"] == summary["decision_id"]
        ]
        if not decision_failures:
            continue
        lines.extend(_render_decision_failures(summary, decision_failures))
    return lines


def _render_decision_failures(
    summary: dict[str, Any], decision_failures: list[dict[str, Any]]
) -> list[str]:
    lines = [
        f"### `{summary['decision_id']}`",
        f"- Decision robustness: `{summary['matched']}/{summary['total']}` variants passed ({summary['pass_rate']}%)",
        "- Review buckets: `query_quality`, `routing_design`, `validator_quality`",
    ]
    for result in decision_failures:
        lines.append(
            f"- Variant `{result['variant_id']}` expected `{result['expected_decision']}` but got `{result['actual_decision'] or '(none)'}`"
        )
        lines.append(f"Query: `{result['query']}`")
        if result.get("notes"):
            lines.append(f"Notes: {result['notes']}")
        lines.append(
            f"Matched signals: `{flatten_signal_summary(result.get('matched_signals', {}))}`"
        )
        lines.append(
            f"Used signals: `{flatten_signal_summary(result.get('used_signals', {}))}`"
        )
    lines.append("")
    return lines


def flatten_signal_summary(signals: dict[str, Any]) -> str:
    if not isinstance(signals, dict):
        return ""
    parts = []
    for key, value in signals.items():
        if isinstance(value, list) and value:
            parts.append(f"{key}={','.join(str(item) for item in value)}")
    return "; ".join(parts)
