"""Check an operator's public-gateway receipt against the proposed count branch."""

import json
import re
from pathlib import Path

if __package__:
    from .headroom import unique_fields
    from .loop_evidence import check, digest
    from .semantic_depth import check_probe
else:
    from headroom import unique_fields
    from loop_evidence import check, digest
    from semantic_depth import check_probe


def fingerprint(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def actual_count(roles, feature):
    known = ("system", "developer", "assistant", "user", "tool")
    if not isinstance(roles, list) or not roles or any(r not in known for r in roles):
        return None
    if feature == "tool_results":
        return roles.count("tool")
    developers = roles.count("developer")
    return len(roles) - max(0, developers - 1)


def required_counts(lower, upper, feature):
    minimum = 1 if feature == "all_messages" else 0
    counts = set()
    if lower > minimum:
        counts.update((lower - 1, lower))
    if upper is not None and upper >= minimum:
        counts.update((upper, upper + 1))
    check(counts, "a public boundary must be observable")
    return counts


def assess_probe(
    path,
    *,
    candidate_sha,
    predicate,
    public_model,
    fast,
    strong,
    backend_aliases,
    require_context_fallback=False,
    require_image_refusal=True,
    require_output_refusal=True,
):
    """Attest only to the checks recorded by the identified external probe producer."""
    check(fingerprint(candidate_sha), "expected candidate SHA-256")
    feature = predicate["feature"]
    semantic = feature == "semantic_depth"
    boundary = set()
    if not semantic:
        lower, upper = predicate["lower"], predicate["upper"]
        check(
            feature in ("tool_results", "all_messages")
            and type(lower) is int
            and lower >= 0
            and (upper is None or (type(upper) is int and lower <= upper)),
            "expected a count predicate with integral bounds",
        )
        boundary = required_counts(lower, upper, feature)
    raw = Path(path).read_bytes()
    receipt = json.loads(raw, object_pairs_hook=unique_fields)
    check(isinstance(receipt, dict), "expected a public probe receipt object")
    issues = []

    def issue(case, field, reason, severity="failed"):
        issues.append(
            {"case": case, "field": field, "reason": reason, "severity": severity}
        )

    def expect(case, row, field, value):
        if field not in row or row[field] is None:
            issue(case, field, "missing evidence", "incomplete")
        elif type(row[field]) is not type(value) or row[field] != value:
            issue(case, field, f"expected {value!r}, received {row[field]!r}")

    kind = receipt.get("kind")
    if kind not in ("real", "fixture"):
        issue(
            "receipt",
            "kind",
            "expected a real or fixture mode",
            "failed" if kind else "incomplete",
        )
    else:
        expect("receipt", receipt, "backend_mode", kind)
    expect("receipt", receipt, "passed", True)
    expect("receipt", receipt, "source_sha", candidate_sha)
    if semantic:
        expect("receipt", receipt, "parent_source_sha", predicate["parent_sha"])
    expect("receipt", receipt, "public_model", public_model)
    if not fingerprint(receipt.get("projected_sha")):
        issue(
            "receipt",
            "projected_sha",
            "missing or malformed deployed projection hash",
            "incomplete",
        )
    fast_lane = predicate.get("decision")
    if not semantic and not fast_lane:
        issue(
            "receipt",
            "decision",
            "candidate fast decision is unidentified",
            "incomplete",
        )
    items = receipt.get("probes")
    if not isinstance(items, list):
        issue("receipt", "probes", "no per-request observations", "incomplete")
        items = []
    by_case, body_hashes, tested = {}, set(), set()
    for row in items:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("case"), str)
            or not row["case"]
        ):
            issue("receipt", "case", "missing probe name", "incomplete")
            continue
        case = row["case"]
        if case in by_case:
            issue(case, "case", "duplicate case")
        by_case[case] = row
        if "public_model" in row:
            expect(case, row, "public_model", public_model)
        sha = row.get("request_sha256")
        if not fingerprint(sha):
            issue(
                case,
                "request_sha256",
                "missing or malformed request identity",
                "incomplete",
            )
        elif sha in body_hashes:
            issue(case, "request_sha256", "same wire request counted more than once")
        else:
            body_hashes.add(sha)

        rejection = {
            "unsupported-image": (503, "no_eligible_model"),
            "output-exceeds-both": (400, "max_output_tokens_exceeded"),
        }.get(case)
        if rejection:
            expect(case, row, "http", rejection[0])
            expect(case, row, "code", rejection[1])
            for field in ("selected", "returned"):
                if field not in row:
                    issue(
                        case,
                        field,
                        "missing backend observation for refusal",
                        "incomplete",
                    )
                elif row[field]:
                    issue(case, field, "refused request reports a backend")
            continue

        if semantic:
            if actual_count(row.get("roles"), "all_messages") is None:
                issue(
                    case, "roles", "missing or unsupported role sequence", "incomplete"
                )
            check_probe(case, row, predicate, fast, backend_aliases, issue, expect)
            if case in ("tool-call", "tool-result"):
                expect(case, row, "tool_ok", True)
            continue
        count = actual_count(row.get("roles"), feature)
        if count is None:
            issue(case, "roles", "missing or unsupported role sequence", "incomplete")
            continue
        if (
            feature == "all_messages"
            and predicate.get("skip_user_messages_matching")
            and row["roles"].count("user") > 1
        ):
            issue(
                case,
                "roles",
                "filtered user count requires message content",
                "incomplete",
            )
            continue
        match = re.fullmatch(r"count-(\d+)", case)
        if match:
            if int(match.group(1)) != count:
                issue(case, "roles", "count label differs from visible roles")
            else:
                tested.add(count)
        inside = count >= lower and (upper is None or count <= upper)
        # The deliberately oversized input exercises the fallback inside the fast lane.
        if case == "long-input" and not inside:
            issue(case, "roles", "oversized input does not exercise the fast branch")
        model = fast if inside and case != "long-input" else strong
        expect(case, row, "http", 200)
        expect(case, row, "selected", model)
        if row.get("returned") not in (
            model,
            backend_aliases.get(model, model),
        ) or not row.get("returned"):
            issue(
                case,
                "returned",
                "missing or different returned backend",
                "incomplete" if not row.get("returned") else "failed",
            )
        if inside:
            if fast_lane:
                expect(case, row, "decision", fast_lane)
        elif predicate.get("fallback_decision"):
            expect(case, row, "decision", predicate["fallback_decision"])
        elif not row.get("decision"):
            issue(case, "decision", "missing selected decision", "incomplete")
        if case in ("tool-call", "tool-result"):
            expect(case, row, "tool_ok", True)

    for n in sorted(boundary - tested):
        issue(f"count-{n}", "roles", "missing distinct boundary request", "incomplete")
    if semantic:
        for case in (
            "depth-changed",
            "depth-protected",
            "depth-overlap",
            "depth-missing",
        ):
            if case not in by_case:
                issue(case, "case", "missing semantic routing check", "incomplete")
    if require_image_refusal and "unsupported-image" not in by_case:
        issue(
            "unsupported-image", "case", "missing unsupported-input probe", "incomplete"
        )
    if require_output_refusal and "output-exceeds-both" not in by_case:
        issue(
            "output-exceeds-both",
            "case",
            "missing oversized-output refusal",
            "incomplete",
        )
    if require_context_fallback and "long-input" not in by_case:
        issue(
            "long-input",
            "case",
            "missing context fallback on the fast branch",
            "incomplete",
        )
    tool_cases = ("tool-call", "tool-result")
    tool_pair_required = kind == "real" or any(case in by_case for case in tool_cases)
    if tool_pair_required:
        for case in tool_cases:
            if case not in by_case:
                issue(case, "case", "missing tool round trip", "incomplete")
    if all(case in by_case for case in tool_cases):
        start = by_case["tool-call"].get("roles")
        end = by_case["tool-result"].get("roles")
        if (
            isinstance(start, list)
            and isinstance(end, list)
            and end != [*start, "assistant", "tool"]
        ):
            issue(
                "tool-result",
                "roles",
                "tool response does not extend the original call history",
            )
        if not semantic and not any(
            by_case[case].get("selected") == fast for case in tool_cases
        ):
            issue(
                "tool-call",
                "selected",
                "tool round trip never exercised the fast model",
            )
    status = f"{kind}_pass"
    if issues:
        status = (
            "failed" if any(i["severity"] == "failed" for i in issues) else "incomplete"
        )
    return {
        "status": status,
        "issues": issues,
        "receipt_sha256": digest(raw),
        "source_sha": receipt.get("source_sha"),
        "projected_sha": receipt.get("projected_sha"),
        "kind": kind,
        "public_model": public_model,
        "tested_boundary_counts": sorted(tested),
        "context_fallback_observed": "long-input" in by_case
        and not any(i["case"] == "long-input" for i in issues),
        "tool_roundtrip_verified": all(case in by_case for case in tool_cases)
        and not issues,
        "scope": "Consistency of recorded requests and responses in the supplied operator receipt.",
    }
