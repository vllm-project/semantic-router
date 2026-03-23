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
    lines.extend(_render_trace_quality_section(after_results))
    lines.extend(_render_root_cause_section(after_results))
    lines.extend(_render_variant_section(after_results))
    lines.extend(_render_review_queue(decision_summaries, after_results))
    if pre_eval and post_eval:
        lines.extend(_render_trajectory_delta(pre_eval, post_eval))
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
    lines = [
        f"- {label} success: `{evaluation['matched']}/{evaluation['total']}` ({evaluation['success_rate']}%)",
        f"- {label} decision coverage: `{evaluation['matched_decisions']}/{evaluation['total_decisions']}` ({evaluation['decision_success_rate']}%)",
    ]
    if "hybrid_reward" in evaluation:
        lines.append(
            f"- {label} hybrid reward: `{evaluation['hybrid_reward']}` "
            f"(trace quality: `{evaluation.get('avg_trace_quality', 'N/A')}`)"
        )
    fragile_count = evaluation.get("fragile_match_count", 0)
    if fragile_count:
        lines.append(f"- {label} fragile matches: `{fragile_count}`")
    return lines


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


def _render_trace_quality_section(results: list[dict[str, Any]]) -> list[str]:
    fragile = [
        r
        for r in results
        if r["matched"]
        and r.get("trace_quality", {}).get("trace_quality", 0) < 0.6
    ]
    lines = ["## Trace Quality", ""]
    if fragile:
        lines.append(
            f"**{len(fragile)} fragile matches** (correct decision, low trace quality):"
        )
        lines.append("")
        lines.append(
            "| Probe | Decision | Trace Quality | Signal Dominance | Avg Confidence |"
        )
        lines.append("|---|---|---|---|---|")
        for r in fragile:
            tq = r.get("trace_quality", {})
            lines.append(
                f"| `{r['id']}` | `{r['actual_decision']}` | "
                f"`{tq.get('trace_quality', 'N/A')}` | "
                f"`{tq.get('signal_dominance', 'N/A')}` | "
                f"`{tq.get('avg_confidence', 'N/A')}` |"
            )
        lines.append("")
    else:
        lines.append(
            "No fragile matches detected. All passing probes have clean traces."
        )
        lines.append("")
    return lines


def _render_root_cause_section(results: list[dict[str, Any]]) -> list[str]:
    failing = [r for r in results if not r["matched"]]
    if not failing:
        return []
    lines = ["## Automated Root-Cause Classification", ""]
    lines.append("| Probe | Expected | Actual | Root Cause | Detail |")
    lines.append("|---|---|---|---|---|")
    for r in failing:
        rc = r.get("root_cause_classification", {})
        lines.append(
            f"| `{r['id']}` | `{r['expected_decision']}` | "
            f"`{r.get('actual_decision', '(none)')}` | "
            f"`{rc.get('root_cause', 'unknown')}` | "
            f"{rc.get('detail', '')} |"
        )
    lines.append("")
    return lines


def _render_trajectory_delta(
    pre_eval: dict[str, Any], post_eval: dict[str, Any]
) -> list[str]:
    delta_sr = post_eval.get("success_rate", 0) - pre_eval.get("success_rate", 0)
    delta_hr = post_eval.get("hybrid_reward", 0) - pre_eval.get("hybrid_reward", 0)
    delta_tq = post_eval.get("avg_trace_quality", 0) - pre_eval.get(
        "avg_trace_quality", 0
    )
    lines = [
        "## Trajectory Delta",
        "",
        f"| Metric | Pre | Post | Delta |",
        f"|---|---|---|---|",
        f"| Success rate | `{pre_eval.get('success_rate', 0)}%` | `{post_eval.get('success_rate', 0)}%` | `{delta_sr:+.1f}%` |",
        f"| Hybrid reward | `{pre_eval.get('hybrid_reward', 0)}` | `{post_eval.get('hybrid_reward', 0)}` | `{delta_hr:+.4f}` |",
        f"| Avg trace quality | `{pre_eval.get('avg_trace_quality', 0)}` | `{post_eval.get('avg_trace_quality', 0)}` | `{delta_tq:+.4f}` |",
        "",
    ]
    return lines


def flatten_signal_summary(signals: dict[str, Any]) -> str:
    if not isinstance(signals, dict):
        return ""
    parts = []
    for key, value in signals.items():
        if isinstance(value, list) and value:
            parts.append(f"{key}={','.join(str(item) for item in value)}")
    return "; ".join(parts)
