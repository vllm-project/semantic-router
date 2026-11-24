"""Measure a tool-window predicate on the fixed prefixes of one development arm."""

import json
from collections import Counter
from pathlib import Path

import yaml

if __package__:
    from .headroom import unique_fields
    from .loop_evidence import check, digest, sha
else:
    from headroom import unique_fields
    from loop_evidence import check, digest, sha


def add_arguments(parser):
    parser.add_argument(
        "--prefixes",
        required=True,
        help="development measurement supplying the requests",
    )
    parser.add_argument(
        "--base", required=True, help="development parent of the window edit"
    )
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument(
        "--schedule-card",
        type=Path,
        action="append",
        default=[],
        help="completed development study releasing assigned evidence",
    )
    parser.add_argument(
        "--upper", type=int, required=True, help="candidate inclusive upper bound"
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="exact configuration emitted by propose",
    )


def visible_tool_count(request):
    # Identify the wire shape on the hash-pinned dispatch. Mutable provider.json
    # can contain secrets and is not an attestation of the historical protocol.
    wire_fields = request.get("wire_fields")
    if (
        not isinstance(wire_fields, list)
        or not all(isinstance(key, str) for key in wire_fields)
        or not {"messages", "model"}.issubset(wire_fields)
        or {"input", "anthropic_version", "system"}.intersection(wire_fields)
    ):
        return None, "unsupported_or_unidentifiable_wire_protocol"
    roles = request.get("roles")
    if (
        roles is None
        or not request.get("messages_sha256")
        or not request.get("last_role")
    ):
        return None, "missing_role_sequence_or_identity"
    sha(request["messages_sha256"])
    check(
        isinstance(roles, list)
        and roles
        and all(isinstance(role, str) for role in roles)
        and roles[-1] == request["last_role"],
        "inconsistent role sequence for dispatch",
    )
    if not all(
        role in {"system", "developer", "user", "assistant", "tool"} for role in roles
    ):
        return None, "unsupported_message_role"
    # Both Chat Completions and Anthropic Messages accept model/messages and
    # max_tokens. Chat-specific roles or fields identify uncapped tool turns.
    if not (
        {"max_completion_tokens", "response_format", "stream_options"}.intersection(
            wire_fields
        )
        or {"system", "developer", "tool"}.intersection(roles)
    ):
        return None, "unsupported_or_unidentifiable_wire_protocol"
    # OpenAI Chat decodes each tool-role message as one tool result. Router's
    # assistant_tool_cycle count reads ToolResultCount for this wire protocol.
    return roles.count("tool"), None


def task_signature(raw, public, action):
    requests, replies = {}, {}
    for line in raw.splitlines():
        row = json.loads(line, object_pairs_hook=unique_fields)
        event = row["event"]
        if event not in {"dispatch", "response"}:
            continue
        index = requests if event == "dispatch" else replies
        call = row["call"]
        check(
            type(call) is int and call > 0 and call not in index,
            "invalid or duplicate call ID",
        )
        index[call] = row
    check(requests and requests.keys() == replies.keys(), "unpaired request receipts")
    before, after, changed = [], [], []
    missing = {}
    for call, request in sorted(requests.items()):
        check(request["model"] == public, "prefix dispatch public model mismatch")
        count, reason = visible_tool_count(request)
        if reason:
            missing[call] = reason
            continue
        old = action["lower"] <= count <= action["before_upper"]
        new = action["lower"] <= count <= action["after_upper"]
        if old:
            before.append(call)
        if new:
            after.append(call)
        if old != new:
            reply = replies[call]
            changed.append(
                {
                    "call": call,
                    "messages_sha256": request["messages_sha256"],
                    "visible_tool_results": count,
                    "before_hit": old,
                    "after_hit": new,
                    "observed_source_decision": reply.get("selected_decision"),
                    "observed_source_model": reply.get("selected_model"),
                }
            )
    added = len(set(after).difference(before))
    unknown = len(missing)
    return {
        "dispatches": len(requests),
        "observed": len(requests) - unknown,
        "unknown": unknown,
        "before_hits": len(before),
        "after_hits": len(after),
        "changed": len(changed),
        "added": added,
        "removed": len(changed) - added,
        "missing_by_reason": dict(Counter(missing.values())),
        "missing_calls": missing,
        "changed_calls": changed,
        "before_hit_calls": before,
        "after_hit_calls": after,
    }


def prefix_receipts(prefixes):
    seen = set()
    for name, fingerprint in prefixes["evidence"].items():
        path = Path(name)
        if path.name != "requests.jsonl":
            continue
        task = path.parents[2].name
        check(
            path.parent.name == "main"
            and path.parents[1].name == prefixes["arm"]
            and path.parents[3].name == "runs"
            and task in prefixes["cases"]
            and task not in seen,
            "receipt does not belong to exactly one declared prefix task and arm",
        )
        seen.add(task)
        raw = path.read_bytes()
        check(digest(raw) == fingerprint, "prefix receipts changed")
        yield task, raw, fingerprint
    check(seen == prefixes["cases"].keys(), "missing prefix task receipts")


def signature(ledger, args):
    # The CLI imports this module to register arguments. Resolve its evidence
    # helpers when handling the command, after the loop module is initialized.
    if __package__:
        from .loop import baseline_config, comparable, released_assignments
        from .planner import independent_dev_row
        from .window_action import apply_window, load_space
    else:
        from loop import baseline_config, comparable, released_assignments
        from planner import independent_dev_row
        from window_action import apply_window, load_space

    base = ledger.measure(args.base)
    prefixes = ledger.measure(args.prefixes)
    check(
        base["split"] == prefixes["split"] == "dev",
        "signature requires two development measurements",
    )
    release_cards = getattr(args, "schedule_card", ())
    released = released_assignments(ledger, release_cards)
    independent_dev_row(base, released)
    if args.base != args.prefixes:
        independent_dev_row(prefixes, released)
    comparable(base, prefixes)
    space_bytes = args.space.read_bytes()
    candidate, action = apply_window(
        baseline_config(base),
        load_space(space_bytes),
        public_model=base["public_model"],
        config_sha256=base["config_sha256"],
        upper=args.upper,
    )
    raw_candidate = args.candidate.read_bytes()
    expected = yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True).encode(
        "utf-8"
    )
    check(
        raw_candidate == expected, "candidate bytes do not match the declared proposal"
    )

    by_task = {}
    for task, raw, fingerprint in prefix_receipts(prefixes):
        summary = task_signature(raw, prefixes["public_model"], action)
        by_task[task] = {
            "requests_sha256": fingerprint,
            "openai_chat_client": "unsupported_or_unidentifiable_wire_protocol"
            not in summary["missing_by_reason"],
            **summary,
        }
    totals = {
        key: sum(task[key] for task in by_task.values())
        for key in (
            "dispatches",
            "observed",
            "unknown",
            "before_hits",
            "after_hits",
            "changed",
            "added",
            "removed",
        )
    }
    return {
        "schema": "recipe-lab-window-signature-v1",
        "based_on": base["record_id"],
        "prefixes": prefixes["record_id"],
        "prefix_arm": prefixes["arm"],
        "base_config_sha256": base["config_sha256"],
        "prefix_config_sha256": prefixes["config_sha256"],
        "candidate_sha256": digest(raw_candidate),
        "space_sha256": digest(space_bytes),
        **(
            {
                "release_cards": [
                    {"file": str(path.resolve()), "sha256": digest(path.read_bytes())}
                    for path in release_cards
                ]
            }
            if release_cards
            else {}
        ),
        "action": action,
        "by_task": by_task,
        "total": totals,
        "observed_behavior": "window_predicate_on_fixed_development_requests",
        "candidate_route_evidence": {
            "status": "unobserved",
            "requires": [
                "other rule conditions and priorities",
                "request and backend eligibility",
                "effective generation and session settings",
                "actual gateway decision and response",
            ],
        },
    }
