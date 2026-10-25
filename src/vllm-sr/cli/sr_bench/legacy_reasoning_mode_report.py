from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

from .legacy_reasoning_modes import FIELD, PROTOCOL
from .transport import normalize_usage

if TYPE_CHECKING:
    from .store import Store


def _resources(calls: list[dict[str, Any]], total: int) -> dict[str, Any]:
    complete = [
        call
        for call in calls
        if call["status"] == "completed" and call.get("output_complete") is True
    ]
    measured: list[tuple[dict[str, int], float]] = []
    reasoning: list[int] = []
    for call in complete:
        raw = call.get("raw_usage")
        usage = normalize_usage(raw) if isinstance(raw, dict) else None
        latency = call.get("latency_s")
        if (
            usage is not None
            and type(latency) in {int, float}
            and math.isfinite(latency)
            and latency >= 0
        ):
            measured.append((usage, latency))
        details = (
            raw.get("completion_tokens_details", {}) if isinstance(raw, dict) else {}
        )
        count = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if type(count) is int and count >= 0:
            reasoning.append(count)
    covered = bool(total) and len(measured) == len(calls) == total
    prompt = sum(
        usage["input_tokens"]
        + usage["cached_input_tokens"]
        + usage["cache_write_tokens"]
        for usage, _ in measured
    )
    output = sum(usage["output_tokens"] for usage, _ in measured)
    seconds = sum(latency for _, latency in measured)
    return {
        "endpoint_usage_basis": "Provider-reported endpoint prompt/completion tokens; excludes Router child-call aggregate usage.",
        "resource_population": "All planned cases require one completed, untruncated subject call with endpoint usage and request latency.",
        "measured_calls": len(measured),
        "measurement_coverage_complete": covered,
        "avg_response_time_s": seconds / total if covered else None,
        "avg_prompt_tokens": prompt / total if covered else None,
        "avg_completion_tokens": output / total if covered else None,
        "completion_prompt_ratio": output / prompt if covered and prompt else None,
        "time_per_completion_token_ms": (
            1000 * seconds / output if covered and output else None
        ),
        "reasoning_usage_reported_calls": len(reasoning),
        "reported_reasoning_tokens": (
            sum(reasoning) if covered and len(reasoning) == total else None
        ),
    }


def make_condition_report(
    store: Store,
    run_id: str,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    from .report import metric  # noqa: PLC0415 - report projection cycle

    section = manifest[FIELD]
    subjects = {
        call["id"]: {
            **call,
            "raw_usage": store.call(run_id, call["id"]).get("raw_usage"),
        }
        for call in calls
        if call["role"] == "subject"
    }
    rows: list[dict[str, Any]] = []
    for condition in section["conditions"]:
        target = condition["target"]
        planned = condition["status"] == "planned"
        prefix = "xc" if condition["mode"] == "XC" else "plain"
        ids = (
            {member[prefix + "_id"] for member in section["members"]}
            if planned
            else set()
        )
        selected_results = [
            row
            for row in results
            if row["target_id"] == target["id"] and row["case_id"] in ids
        ]
        selected_calls = [
            call
            for call in subjects.values()
            if call["target_id"] == target["id"] and call["case_id"] in ids
        ]
        categories: list[dict[str, Any]] = []
        for category in sorted({member["stratum"] for member in section["members"]}):
            category_ids = {
                member[prefix + "_id"]
                for member in section["members"]
                if member["stratum"] == category
            } & ids
            category_calls = [
                call for call in selected_calls if call["case_id"] in category_ids
            ]
            categories.append(
                {
                    "category": category,
                    **metric(
                        target_id=target["id"],
                        results=[
                            row
                            for row in selected_results
                            if row["case_id"] in category_ids
                        ],
                        calls=category_calls,
                        total=len(category_ids),
                        planned_case_ids=category_ids,
                    ),
                    **_resources(category_calls, len(category_ids)),
                }
            )
        rows.append(
            {
                "requested_mode": condition["requested_mode"],
                "mode": condition["mode"],
                "target_id": target["id"],
                "requested_model": target["model"],
                "group": condition["group"],
                "status": condition["status"],
                "skip_reason": condition["reason"],
                "requested_cases": len(section["members"]),
                "controls": condition["controls"],
                "control_evidence": "Requested wire settings, not proof that the backend enabled or disabled reasoning.",
                "prompt_condition": (
                    "native_explicit_cot" if prefix == "xc" else "native_plain"
                ),
                "summary": {
                    **metric(
                        target_id=target["id"],
                        results=selected_results,
                        calls=selected_calls,
                        total=len(ids),
                        planned_case_ids=ids,
                    ),
                    **_resources(selected_calls, len(ids)),
                },
                "categories": categories,
                "finish_reasons": dict(
                    Counter(
                        call.get("finish_reason") or "unknown"
                        for call in selected_calls
                    )
                ),
            }
        )
    deltas: list[dict[str, Any]] = []
    for baseline in rows:
        if baseline["mode"] != "NR":
            continue
        for row in rows:
            if row["group"] != baseline["group"] or row["mode"] not in {
                "NR_REASONING",
                "AR",
            }:
                continue
            complete = all(item["summary"]["complete"] for item in (baseline, row))
            deltas.append(
                {
                    "baseline_target_id": baseline["target_id"],
                    "target_id": row["target_id"],
                    "accuracy_delta": (
                        row["summary"]["accuracy"] - baseline["summary"]["accuracy"]
                        if complete
                        else None
                    ),
                    "basis": "Descriptive delivered accuracy on identical plain cases; not a causal estimate.",
                }
            )
    return {
        "protocol": PROTOCOL,
        "selection_sha256": section["selection_sha256"],
        "conditions": rows,
        "plain_quality_deltas": deltas,
        "limitations": [
            "Explicit-CoT changes the input and can include a supplied explanation. It is not an identical-input reasoning-toggle comparison.",
            "Streaming, visible-final-only grading and output-limit failures differ from native runner response handling.",
            "Sequential conditions do not establish matched cache state, a latency/cost improvement, or a production recommendation.",
        ],
    }
