"""Read back one prospective development decision and the evidence a proposer can use."""

import re
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .loop_evidence import check, component, digest, sha
    from .schedule import canonical, decision, parse, protocol, replay_attempts
else:
    from loop_evidence import check, component, digest, sha
    from schedule import canonical, decision, parse, protocol, replay_attempts


def pinned_episode_file(row, task, filename):
    check(row["split"] == "dev", "observations require development measurements")
    suffix = ("runs", task, row["arm"], "main", filename)
    files = [Path(name) for name in row["evidence"] if Path(name).parts[-5:] == suffix]
    check(len(files) == 1, f"expected one development {filename}")
    raw = files[0].read_bytes()
    check(digest(raw) == row["evidence"][str(files[0])], f"changed {filename}")
    return raw


def requests_and_terminal(row, task):
    raw = pinned_episode_file(row, task, "requests.jsonl")
    dispatch, response = {}, {}
    for line in raw.splitlines():
        event = parse(line)
        if event["event"] in ("dispatch", "response"):
            bucket = dispatch if event["event"] == "dispatch" else response
            check(event["call"] not in bucket, "duplicate call receipt")
            bucket[event["call"]] = event
    check(dispatch and dispatch.keys() == response.keys(), "unpaired call receipt")
    result = parse(pinned_episode_file(row, task, "result.json"))
    outcome = row["cases"][task]
    check(
        result["termination"] == outcome["termination"]
        and result["grade"]["correct"] is outcome["correct"],
        "development result disagrees with measurement",
    )
    call = next(reversed(dispatch))
    params = dispatch[call].get("params")
    reply = response[call]
    complete = reply.get("http") == 200 and reply.get("done") is True
    usage = reply.get("usage") if complete else None
    details = (
        usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    )
    finish = reply.get("finish") if complete else None
    terminal = {
        "termination": outcome["termination"],
        "last_call": call,
        "finish_reason": (
            finish
            if finish
            in ("stop", "length", "tool_calls", "function_call", "content_filter")
            else None
        ),
        "explicit_request_limits": {
            key: params[key]
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if isinstance(params, dict)
            and type(params.get(key)) is int
            and params[key] > 0
        },
        "completion_tokens": token_count(usage, "completion_tokens"),
        "reasoning_tokens": token_count(details, "reasoning_tokens"),
        "upstream_request_limits": None,
    }
    return dispatch, response, terminal


def token_count(obj, key):
    value = obj.get(key) if isinstance(obj, dict) else None
    return value if type(value) is int and value >= 0 else None


def output_limit_symptom(strict, terminal):
    return strict is False and terminal["termination"] == "native_output_limit"


def route_receipt(ledger, item):
    row = ledger.measure(item["evidence"]["measurement"])
    dispatch, response, terminal = requests_and_terminal(row, item["task"])
    selected = Counter()
    by_body = defaultdict(list)
    for call, req in dispatch.items():
        reply = response[call]
        model = None
        if (
            reply.get("http") == 200
            and reply.get("done") is True
            and reply.get("returned_model")
        ):
            model = reply.get("selected_model")
        if model:
            selected[model] += 1
        body = req.get("body_sha256_in")
        if (
            isinstance(body, str)
            and re.fullmatch("[0-9a-f]{64}", body)
            and req.get("body_sha256_sent") == body
        ):
            by_body[body].append(model)
    return (
        {
            "calls": len(dispatch),
            "observed_models": dict(selected),
            "unknown_model_calls": len(dispatch) - sum(selected.values()),
        },
        by_body,
        terminal,
    )


def actual_observations(ledger, history, tasks):
    results = {task: {"attempts": [], "matched_requests": []} for task in tasks}
    routes = defaultdict(dict)
    for item in history:
        task, rep, role = item["task"], item["repeat"], item["role"]
        fields = (
            "assignment_id",
            "role",
            "repeat",
            "valid",
            "seconds",
            "timing_source",
            "episode_sha256",
        )
        row = {key: item[key] for key in fields}
        if item["valid"]:
            observed, bodies, terminal = route_receipt(ledger, item)
            row.update(
                strict=item["strict"],
                measurement=item["evidence"]["measurement"],
                routing=observed,
                terminal=terminal,
            )
            routes[task, rep][role] = bodies
        else:
            row["invalid_reason"] = item["reason"]
        results[task]["attempts"].append(row)
    for (task, rep), roles in sorted(routes.items()):
        if roles.keys() != {"baseline", "candidate"}:
            continue
        left, right = roles["baseline"], roles["candidate"]
        shared = pairs = changes = 0
        for body in sorted(left.keys() & right.keys()):
            shared += 1
            if (
                len(left[body]) == len(right[body]) == 1
                and left[body][0]
                and right[body][0]
            ):
                pairs += 1
                changes += int(left[body][0] != right[body][0])
        results[task]["matched_requests"].append(
            {
                "repeat": rep,
                "shared_body_hashes": shared,
                "comparable_requests": pairs,
                "different_selected_models": changes if pairs else None,
            }
        )
    return results


REOPEN = {
    "advance": "Confirm quality and elapsed time on independent tasks before adoption.",
    "revise": "Use observed decisions and paired times to choose a new edit or an informative repeat.",
    "drop": "Reopen with a changed guard mechanism or a repeat that can discriminate the proposed explanation.",
    "no observed target predicate change": "Collect a relevant development prefix or test a different action.",
    "episode budget exhausted before complete pairs": "Register more budget for the missing pairs.",
    "the fresh guard baseline did not reproduce": "Reproduce guard baseline success before retesting the edit.",
    "opposite paired quality outcomes at equal pass counts": "Repeat the affected tasks before using latency to advance the candidate.",
}


def summarize(ledger, card_path):
    raw_card = card_path.read_bytes()
    study = component(parse(raw_card)["study"])
    state_path = ledger.path.parent / "schedules" / study / "state.json"
    state_bytes = state_path.read_bytes() if state_path.is_file() else None
    state = parse(state_bytes) if state_bytes is not None else None
    if state is not None:
        check(
            state["card_sha256"] == digest(raw_card),
            "experiment card changed after enrollment",
        )
    card, card_hash, signature, parent = protocol(ledger, card_path)
    check(card_hash == digest(raw_card), "experiment card changed during readback")
    tasks = (card["guard_task"], card["target_task"])
    result = {
        "schema": "recipe-lab-decision-summary-v1",
        "study": study,
        "split": "dev",
        "card_sha256": card_hash,
        "state_sha256": digest(state_bytes) if state_bytes is not None else None,
        "parent": parent["record_id"],
        "parent_config_sha256": signature["base_config_sha256"],
        "task_controls_sha256": {
            t: digest(canonical(parent["cases"][t]["controls"]).encode()) for t in tasks
        },
        "candidate_sha256": signature["candidate_sha256"],
        "prefixes": signature["prefixes"],
        "signature_sha256": card["signature_sha256"],
        "hypotheses": card["hypotheses"],
        "prediction": card["prediction"],
        "action": signature["action"],
        "guard_task": tasks[0],
        "target_task": tasks[1],
        "fixed_prefixes": {
            t: {
                k: signature["by_task"][t][k]
                for k in ("observed", "unknown", "changed")
            }
            for t in tasks
        },
    }
    if state is None or "enrollment" not in state:
        check(
            not state or not state["attempts"],
            "legacy nonempty schedule lacks prospective assignments",
        )
        result.update(
            decision="not_enrolled",
            complete=False,
            next_action="Enroll the card before dispatch.",
        )
    else:
        sha(state["enrollment"])
        history = replay_attempts(ledger, card, signature, parent, state)
        plan = decision(card, signature, history)
        status = plan["status"]
        result.update(decision=status, complete=status != "run")
        result["cost"] = {
            k: plan[k]
            for k in (
                "spent_episodes",
                "budget_episodes",
                "invalid_attempts",
                "valid_agent_seconds",
                "invalid_attempt_wall_seconds",
            )
        }
        if status == "run":
            result["next_action"] = (
                "Finish the scheduled paired development comparison."
            )
        else:
            result.update(
                reason=plan["reason"],
                next_action=plan.get("consequence", plan["reason"]),
            )
            if "repeat_tasks" in plan:
                result["repeat_tasks"] = plan["repeat_tasks"]
            result["by_task"] = actual_observations(ledger, history, tasks)
            if any(
                output_limit_symptom(attempt["strict"], attempt["terminal"])
                for stats in result["by_task"].values()
                for attempt in stats["attempts"]
                if attempt["valid"]
            ):
                result["symptoms"] = ["strict_failure_native_output_limit"]
            for task, stats in plan.get("paired_by_task", {}).items():
                result["by_task"][task]["paired"] = {
                    **stats,
                    "joint_seconds_saved": (
                        stats["joint_seconds_saved"]
                        if stats["joint_successes"]
                        else None
                    ),
                }
            result["reopen_when"] = REOPEN[
                plan["reason"] if status == "defer" else status
            ]
            result["unverified"] = [
                "Only equal request-body hashes with one complete model observation per arm are compared within each task and repeat.",
                "Independent trajectories cannot establish that a backend selection caused the task outcome or its duration.",
                "Transfer to new tasks and deployment benefit require independent evaluation.",
            ]
    latest = state_path.read_bytes() if state_path.is_file() else None
    check(
        state_bytes == latest and card_path.read_bytes() == raw_card,
        "schedule changed during readback; retry",
    )
    return result
