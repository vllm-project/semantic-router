"""Replay a proposed fast-lane predicate on the recorded requests of one task."""

import json
import math

if __package__:
    from .headroom import unique_fields
    from .loop_evidence import check, fit_status, is_number
    from .signature import visible_tool_count
else:
    from headroom import unique_fields
    from loop_evidence import check, fit_status, is_number
    from signature import visible_tool_count


def paired_requests(raw, public_model):
    dispatches, responses = {}, {}
    for line in raw.splitlines():
        row = json.loads(line, object_pairs_hook=unique_fields)
        event = row["event"]
        if event not in ("dispatch", "response"):
            continue
        bucket = dispatches if event == "dispatch" else responses
        call = row["call"]
        check(
            type(call) is int and call > 0 and call not in bucket,
            "invalid or duplicate call ID",
        )
        if event == "dispatch":
            check(row["model"] == public_model, "prefix dispatch public model mismatch")
        bucket[call] = row
    check(
        dispatches and dispatches.keys() == responses.keys(),
        "unpaired request receipts",
    )
    for call, request in sorted(dispatches.items()):
        yield call, request, responses[call]


def interval(request, response):
    start, end = request.get("start"), response.get("end")
    if start is None or end is None:
        return None
    check(
        is_number(start) and is_number(end) and start <= end,
        "invalid request time interval",
    )
    return start, end


def union_seconds(spans):
    if any(s is None for s in spans):
        return None
    # A duration alone cannot tell whether two concurrent requests overlap.
    end = -math.inf
    durations = []
    for a, b in sorted(spans):
        durations.append(max(0, b - max(a, end)))
        end = max(end, b)
    return round(math.fsum(durations), 6)


def token_fit(request, response, fast_context, fast_max_output):
    params = request.get("params")
    limit_keys = ("max_tokens", "max_completion_tokens", "max_output_tokens")
    fields = request.get("wire_fields") or []
    limits = (
        [params[k] for k in limit_keys if k in params and k in fields]
        if isinstance(params, dict)
        else []
    )
    requested = None
    if limits and all(type(v) is int and v > 0 and v == limits[0] for v in limits):
        requested = limits[0]
    usage = response.get("usage")
    prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if type(prompt) is not int or prompt < 0:
        prompt = None
    return {
        "status": fit_status(requested, prompt, fast_context, fast_max_output),
        "requested_output_tokens": requested,
        "source_prompt_tokens": prompt,
    }


def source_binding(response, fast, strong, aliases):
    if response.get("http") != 200 or response.get("done") is not True:
        return None
    selected = response.get("selected_model")
    returned = response.get("returned_model")
    if not isinstance(returned, str) or not returned or selected not in (strong, fast):
        return None
    if returned == selected or returned == aliases.get(selected):
        return selected
    return None


def visible_count(request, feature, skip_user_messages_matching):
    tools, reason = visible_tool_count(request)
    if reason or feature == "tool_results":
        return tools, reason
    roles = request["roles"]
    if skip_user_messages_matching and roles.count("user") > 1:
        # RoutingInput can drop a matching user turn, depending on its content.
        # When all match the selector keeps the last one with text.
        return None, "user_message_filter_requires_message_content"
    # The Router's countMessagesByRole counts every system instruction and
    # records developer messages with the single HasDeveloperMessage bit.
    developers = roles.count("developer")
    return len(roles) - max(0, developers - 1), None


def replay_task(
    raw,
    *,
    public_model,
    fast,
    strong,
    candidate,
    fast_context,
    fast_max_output,
    task_seconds,
    backend_aliases=None,
    skip_user_messages_matching=False,
    depth_observations=None,
):
    """Measure observed exposure on a fixed trace, without simulating model answers."""
    feature = candidate["feature"]
    check(
        feature
        in (
            "all_messages",
            "tool_results",
            "fixed_strong",
            "unknown",
            "semantic_depth",
        ),
        "unsupported fast-lane predicate",
    )
    count_feature = feature in ("all_messages", "tool_results")
    if count_feature:
        lower, upper = candidate["lower"], candidate["upper"]
        check(
            type(lower) is int
            and lower >= 0
            and (upper is None or (type(upper) is int and upper >= lower)),
            "invalid predicate bounds",
        )
    check(
        fast != strong and fast and strong, "distinct fast and strong binding required"
    )
    for limit in (fast_context, fast_max_output):
        check(
            limit is None or (type(limit) is int and limit > 0),
            "invalid fast model limits",
        )
    check(
        task_seconds is None or (is_number(task_seconds) and task_seconds > 0),
        "invalid task duration",
    )
    calls, unknown = [], {}
    changed_decisions = []
    spans = {}
    marked = {
        key: []
        for key in (
            "matched_calls",
            "matched_source_fast_calls",
            "matched_source_strong_calls",
            "matched_unknown_source_calls",
            "opportunity_calls",
            "missing_fit_calls",
            "hard_blocked_calls",
            "fit_estimate_exceeded_calls",
            "rate_limited_calls",
            "transport_failure_calls",
            "incomplete_calls",
        )
    }
    for call, request, response in paired_requests(raw, public_model):
        elapsed = interval(request, response)
        spans[call] = elapsed
        http, done = response.get("http"), response.get("done")
        complete = http == 200 and done is True
        if not complete:
            marked["transport_failure_calls"].append(call)
            if http == 429:
                marked["rate_limited_calls"].append(call)
            if http == 200:
                marked["incomplete_calls"].append(call)

        count = None
        matches = False
        if count_feature:
            count, reason = visible_count(request, feature, skip_user_messages_matching)
            if reason:
                unknown[call] = reason
                matches = None
            else:
                matches = count >= lower and (upper is None or count <= upper)
        elif feature == "semantic_depth":
            observation = (depth_observations or {}).get(call)
            if observation is None:
                matches = None
                unknown[call] = "missing_development_depth_observation"
            else:
                score = observation["score"]
                in_band = (
                    score is not None
                    and candidate["before"] <= score < candidate["after"]
                )
                if in_band:
                    changed_decisions.append(call)
                matches = (
                    in_band
                    and not observation["matched_other_decisions"]
                    and observation["fast_eligible"]
                )
                if (
                    matches
                    and token_fit(request, response, fast_context, fast_max_output)[
                        "status"
                    ]
                    == "source_token_estimate_exceeds_context"
                ):
                    matches = False
        elif feature == "unknown":
            unknown[call] = "unsupported_decision_conditions"
            matches = None

        binding = source_binding(response, fast, strong, backend_aliases or {})
        fit = (
            token_fit(request, response, fast_context, fast_max_output)
            if matches
            else None
        )
        if matches:
            marked["matched_calls"].append(call)
            if binding == fast:
                marked["matched_source_fast_calls"].append(call)
            elif binding == strong:
                marked["matched_source_strong_calls"].append(call)
            elif complete:
                marked["matched_unknown_source_calls"].append(call)
            status = fit["status"]
            if status.startswith("requested_output_exceeds_"):
                marked["hard_blocked_calls"].append(call)
            elif binding == strong:
                marked["opportunity_calls"].append(call)
                if status == "unknown":
                    marked["missing_fit_calls"].append(call)
                elif status == "source_token_estimate_exceeds_context":
                    marked["fit_estimate_exceeded_calls"].append(call)
        calls.append(
            {
                "call": call,
                "messages_sha256": request.get("messages_sha256"),
                "count": count,
                **(
                    {"depth_observation": observation}
                    if feature == "semantic_depth"
                    else {}
                ),
                "predicate_hit": matches,
                "http": http,
                "done": done,
                "source_decision": response.get("selected_decision"),
                "source_binding": binding,
                "selected_model": response.get("selected_model"),
                "returned_model": response.get("returned_model"),
                "fast_token_fit": fit,
                "request_seconds": (
                    None if elapsed is None else round(elapsed[1] - elapsed[0], 6)
                ),
            }
        )
    opportunities = marked["opportunity_calls"]
    seconds = union_seconds([spans[i] for i in opportunities])
    if len(unknown) == len(calls):
        seconds = None
    fraction = None
    if task_seconds and seconds is not None and seconds <= task_seconds:
        fraction = seconds / task_seconds
    return {
        "dispatches": len(calls),
        **(
            {"changed_decision_calls": changed_decisions}
            if feature == "semantic_depth"
            else {}
        ),
        **marked,
        "unknown_calls": unknown,
        "calls": calls,
        "opportunity_seconds": seconds,
        "opportunity_fraction": fraction,
        "model_interval_seconds": union_seconds(list(spans.values())),
        "missing_opportunity_times": [i for i in opportunities if spans[i] is None],
        "blockers": candidate.get("blockers", []),
        "actual_candidate_route": "unobserved",
    }
