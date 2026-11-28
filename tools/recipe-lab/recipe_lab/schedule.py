"""Plan and resume a paired, whole-task development comparison."""

import json
import math
import os
import secrets
import tempfile
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from .headroom import unique_fields
    from .loop_evidence import check, check_client, component, digest, sha
    from .signature import signature
else:
    from headroom import unique_fields
    from loop_evidence import check, check_client, component, digest, sha
    from signature import signature


def add_arguments(parser):
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument(
        "--attempt", type=Path, help="import one completed or invalid attempt"
    )


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_fields)


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)


def meaningful(value):
    return isinstance(value, str) and bool(value.strip())


def validated_plan(ledger, card, plan, candidate_path):
    if __package__:
        from .planner import replay_plan
    else:
        from planner import replay_plan
    check(
        plan["schema"] == "recipe-lab-development-plan-v1"
        and plan["based_on"] == card["based_on"],
        "wrong development plan or parent",
    )
    verified = replay_plan(ledger, plan)
    check(
        same_receipt(verified, plan), "plan differs from verified development evidence"
    )
    step = verified["next"]
    target, guard = card["target_task"], card["guard_task"]
    check(
        step["action"] == "plan_paired_development"
        and target == step["task"]
        and guard in step["reproduced_guard_options"],
        "plan requires a verified real probe, ranked target and distinct reproduced guard",
    )
    check(
        digest(candidate_path.read_bytes()) == step["candidate_sha256"],
        "candidate bytes differ from the planned public probe",
    )
    sources = verified["prefix_evidence"][step["candidate"]]
    by_task, cases = {}, {}
    for task in (guard, target):
        rows = [r for r in sources if r["task"] == task]
        # The planner has verified all repeats for this task used identical controls.
        cases[task] = ledger.measure(rows[0]["measurement"])["cases"][task]
        by_task[task] = {
            "observed": sum(r["dispatches"] - len(r["unknown_calls"]) for r in rows),
            "unknown": sum(len(r["unknown_calls"]) for r in rows),
            "changed": sum(len(r["opportunity_calls"]) for r in rows),
        }
    selected = next(
        c for c in verified["space"]["candidates"] if c["id"] == step["candidate"]
    )
    report = {
        "base_config_sha256": verified["space"]["base_config_sha256"],
        "candidate_sha256": step["candidate_sha256"],
        "by_task": by_task,
        "prefixes": list(dict.fromkeys(row["measurement"] for row in sources)),
        "action": {
            "candidate": step["candidate"],
            "predicate": selected["predicate"],
            "coverage": "source strong requests reachable by the candidate predicate",
        },
    }
    return report, {**ledger.measure(card["based_on"]), "cases": cases}


def same_receipt(actual, saved):
    # JSON object keys become strings on disk (including numeric call IDs).
    return canonical(parse(json.dumps(actual))) == canonical(saved)


_validating_cards = ContextVar("recipe_lab_validating_cards", default=frozenset())


def protocol(ledger, path):
    # Standalone development scheduling returns the checked card directly;
    # the cross-method search layer (sub-issue 10) wraps it with
    # search_accounting.historical_card.
    return checked_protocol(ledger, path)


def checked_protocol(ledger, path):
    identity = path.resolve()
    active = _validating_cards.get()
    check(identity not in active, "cyclic development schedule dependency")
    token = _validating_cards.set(active | {identity})
    try:
        return _protocol(ledger, path)
    finally:
        _validating_cards.reset(token)


def _protocol(ledger, path):
    raw = path.read_bytes()
    card = parse(raw)
    check(
        card["schema"] in ("recipe-lab-schedule-v1", "recipe-lab-schedule-v2"),
        "unknown experiment card schema",
    )
    component(card["study"])
    check(
        type(card["budget_episodes"]) is int and card["budget_episodes"] >= 0,
        "budget_episodes must be a nonnegative whole number",
    )
    check(
        type(card["repeats"]) is int and card["repeats"] > 0,
        "repeats must be a positive whole number",
    )
    hypotheses = card["hypotheses"]
    check(
        isinstance(hypotheses, list)
        and len(hypotheses) == 2
        and all(map(meaningful, hypotheses))
        and hypotheses[0].strip() != hypotheses[1].strip(),
        "record two different explanations",
    )
    check(meaningful(card["prediction"]), "record the predicted change")
    check(
        set(card["decisions"]) == {"advance", "revise", "drop"}
        and all(map(meaningful, card["decisions"].values())),
        "record the three decision consequences",
    )
    report_raw = (path.parent / card["signature"]).read_bytes()
    check(
        digest(report_raw) == sha(card["signature_sha256"]), "signature receipt changed"
    )
    report = parse(report_raw)
    if card["schema"] == "recipe-lab-schedule-v2":
        report, parent = validated_plan(
            ledger, card, report, path.parent / card["candidate"]
        )
        return card, digest(raw), report, parent
    releases = []
    for ref in report.get("release_cards", []):
        release = Path(ref["file"])
        check(
            digest(release.read_bytes()) == sha(ref["sha256"]),
            "development release card changed",
        )
        releases.append(release)
    actual = signature(
        ledger,
        SimpleNamespace(
            base=card["based_on"],
            prefixes=report["prefixes"],
            candidate=path.parent / card["candidate"],
            space=path.parent / card["space"],
            upper=report["action"]["after_upper"],
            schedule_card=releases,
        ),
    )
    check(
        same_receipt(actual, report),
        "signature differs from verified development evidence",
    )
    parent = ledger.measure(card["based_on"])
    target, guard = card["target_task"], card["guard_task"]
    check(
        target != guard and {target, guard} <= set(report["by_task"]),
        "need distinct development target and guard",
    )
    check(
        parent["cases"][guard]["correct"] is True,
        "the guard needs a baseline strict success",
    )
    return card, digest(raw), report, parent


def measurements_for(ledger, attempt, parent, config_hash):
    if __package__:
        from .loop import verify
    else:
        from loop import verify
    row = ledger.measure(attempt["measurement"])
    check(row["split"] == "dev", "scheduled measurement must be development evidence")
    verify(row)
    task = attempt["task"]
    check(
        row["public_model"] == parent["public_model"]
        and row["strong_model"] == parent["strong_model"]
        and row["config_sha256"] == config_hash,
        "measurement is not the scheduled development configuration",
    )
    case = row["cases"][task]
    check(
        canonical(case["controls"]) == canonical(parent["cases"][task]["controls"]),
        "task or run controls differ from the frozen parent",
    )
    result_files = [
        Path(p)
        for p in row["evidence"]
        if Path(p).parts[-5:] == ("runs", task, row["arm"], "main", "result.json")
    ]
    check(len(result_files) == 1, "expected one audited per-task run")
    result = result_files[0]
    binding = result.parents[4] / "bindings" / task / (row["arm"] + ".json")
    raw = binding.read_bytes()
    check(
        row["evidence"].get(str(binding)) == digest(raw),
        "missing or changed graded run binding",
    )
    check(
        parse(raw).get("assignment_id") == attempt["assignment_id"],
        "graded run binding lacks the matching prospective assignment",
    )
    files = {
        k: row["evidence"][str(result.parent / k)]
        for k in ("result.json", "requests.jsonl")
    }
    return {
        "valid": True,
        "strict": case["correct"],
        "seconds": case["seconds"],
        "timing_source": "dsh_agent_seconds",
        "run_fingerprints": files,
        "evidence": {"measurement": row["record_id"]},
    }


def invalid_attempt(attempt, parent, config_hash):
    task, arm = component(attempt["task"]), component(attempt["arm"])
    root = Path(attempt["invalid_run"]).resolve() / "dsh"
    folder = root / "runs" / task / arm / "main"
    meta = {}

    def read(file):
        raw = file.read_bytes()
        meta[str(file.relative_to(root))] = digest(raw)
        return parse(raw)

    frozen = read(root / "frozen.json")
    binding = read(root / "bindings" / task / (arm + ".json"))
    controls = parent["cases"][task]["controls"]
    check(
        canonical(frozen["cases"][task]) == canonical(controls["case"]),
        "invalid-run task mismatch",
    )
    assignment = attempt["assignment_id"]
    check(
        binding.get("assignment_id") == assignment,
        "invalid run binding lacks the matching prospective assignment",
    )
    check(
        binding["model"] == parent["public_model"]
        and binding["source_sha256"] == config_hash
        and binding["client_window"] == controls["client_window"],
        "invalid attempt is not from the scheduled deployment",
    )
    sha(binding["active_runtime_hash"])
    start = read(folder / "execution-start.json")
    end = read(folder / "execution-end.json")
    check(
        start["assignment_id"] == end["assignment_id"] == assignment
        and start["controls_sha256"] == digest(canonical(controls).encode())
        and start["clock"] == end["clock"] == "CLOCK_BOOTTIME"
        and meaningful(start["boot_id"])
        and start["boot_id"] == end["boot_id"]
        and type(start["monotonic_ns"]) is int
        and type(end["monotonic_ns"]) is int
        and 0 <= start["monotonic_ns"] < end["monotonic_ns"],
        "invalid attempt lacks matching launch controls and a same-boot execution span",
    )
    runner_exit, judge_exit, reason = (
        end["runner_exit"],
        end["judge_exit"],
        end["status"],
    )
    check(
        type(runner_exit) is int and (judge_exit is None or type(judge_exit) is int),
        "invalid runner or judge exit evidence",
    )
    native = read(folder / "result.json") if (folder / "result.json").is_file() else {}
    if reason == "model_endpoint_error":
        check(native.get("termination") == reason, "missing native endpoint error")
    elif reason == "runner_error":
        check(
            runner_exit != 0
            or native.get("runner_error")
            or native.get("termination") == "dsh_error",
            "missing runner failure evidence",
        )
    elif reason == "grader_error":
        check(
            (judge_exit is not None and judge_exit != 0)
            or (
                judge_exit is None
                and native.get("judge_error") == "official_verifier_timeout"
            ),
            "missing independent grader failure evidence",
        )
        if "judge_exit" in native:
            check(native["judge_exit"] == judge_exit, "conflicting native grader exit")
    else:
        raise ValueError("unsupported invalid-attempt reason")
    if (folder / "protocol.json").is_file():
        proto = read(folder / "protocol.json")
        check(
            proto["model"] == parent["public_model"]
            and proto["arm"] == arm
            and proto["verified_task_packet_sha256"] == meta["frozen.json"]
            and proto["prompt_sha256"] == controls["case"]["instruction_sha256"],
            "invalid attempt uses a different packet or arm",
        )
        for key, val in controls.items():
            if key not in ("case", "client_window"):
                check(
                    canonical(proto[key]) == canonical(val),
                    f"invalid attempt control differs: {key}",
                )
    else:
        check(reason == "runner_error", "native execution protocol missing")
    if (folder / "provider.json").is_file():
        check_client(folder, parent["public_model"], controls["client_window"])
    files = {
        n: digest((folder / n).read_bytes())
        for n in ("result.json", "requests.jsonl")
        if (folder / n).is_file()
    }
    if len(files) != 2:
        files.update(
            {
                n: meta[str((folder / n).relative_to(root))]
                for n in ("execution-start.json", "execution-end.json")
            }
        )
    return {
        "valid": False,
        "reason": reason,
        "seconds": (end["monotonic_ns"] - start["monotonic_ns"]) / 1_000_000_000,
        "timing_source": "driver_attempt_wall_seconds",
        "run_fingerprints": files,
        "evidence": {"invalid_run": str(root.parent), "arm": arm, "file_sha256": meta},
    }


def collect(ledger, attempt, parent, report):
    check(
        "assignment_id" in attempt, "attempt must carry its prospective assignment_id"
    )
    sha(attempt["assignment_id"])
    check(
        attempt["role"] in ("baseline", "candidate") and type(attempt["repeat"]) is int,
        "invalid scheduled role or repeat",
    )
    role = attempt["role"]
    config_hash = (
        report["base_config_sha256"]
        if role == "baseline"
        else report["candidate_sha256"]
    )
    check(
        ("measurement" in attempt) != ("invalid_run" in attempt),
        "supply one audited measurement or one invalid_run with arm",
    )
    result = (
        measurements_for(ledger, attempt, parent, config_hash)
        if "measurement" in attempt
        else invalid_attempt(attempt, parent, config_hash)
    )
    seconds = result["seconds"]
    check(
        type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
        "invalid episode duration",
    )
    return {
        **{k: attempt[k] for k in ("repeat", "task", "role", "assignment_id")},
        **result,
        "config_sha256": config_hash,
        "episode_sha256": digest(canonical(result["run_fingerprints"]).encode()),
    }


def decision(card, report, history):
    cases = {}
    for item in history:
        if item["valid"]:
            cases[item["repeat"], item["task"], item["role"]] = item
    target, guard = card["target_task"], card["guard_task"]
    result = {
        "spent_episodes": len(history),
        "budget_episodes": card["budget_episodes"],
        "invalid_attempts": len(history) - len(cases),
        "seconds_all_attempts": math.fsum(x["seconds"] for x in history),
        "valid_agent_seconds": math.fsum(x["seconds"] for x in history if x["valid"]),
        "invalid_attempt_wall_seconds": math.fsum(
            x["seconds"] for x in history if not x["valid"]
        ),
    }
    if not report["by_task"][target]["changed"]:
        return {
            **result,
            "status": "defer",
            "reason": "no observed target predicate change",
        }
    stats = {
        task: {
            "baseline_passes": 0,
            "candidate_passes": 0,
            "joint_successes": 0,
            "joint_seconds_saved": 0.0,
        }
        for task in (guard, target)
    }
    result["paired_by_task"] = stats
    for rep in range(1, card["repeats"] + 1):
        for task_idx, task in enumerate((guard, target)):
            roles = (
                ("baseline", "candidate")
                if (rep + task_idx) % 2
                else ("candidate", "baseline")
            )
            for role in roles:
                if (rep, task, role) in cases:
                    continue
                if len(history) >= card["budget_episodes"]:
                    return {
                        **result,
                        "status": "defer",
                        "reason": "episode budget exhausted before complete pairs",
                    }
                config_key = (
                    "base_config_sha256" if role == "baseline" else "candidate_sha256"
                )
                return {
                    **result,
                    "status": "run",
                    "next": {
                        "repeat": rep,
                        "task": task,
                        "role": role,
                        "config_sha256": report[config_key],
                    },
                }
            baseline, candidate = [
                cases[rep, task, r] for r in ("baseline", "candidate")
            ]
            stat = stats[task]
            stat["baseline_passes"] += int(baseline["strict"])
            stat["candidate_passes"] += int(candidate["strict"])
            if baseline["strict"] and candidate["strict"]:
                stat["joint_successes"] += 1
                stat["joint_seconds_saved"] += (
                    baseline["seconds"] - candidate["seconds"]
                )
            if (
                rep == card["repeats"]
                and stat["candidate_passes"] < stat["baseline_passes"]
            ):
                return {
                    **result,
                    "status": "drop",
                    "reason": f"strict regression on {task}",
                    "consequence": card["decisions"]["drop"],
                }
    gains = sum(s["candidate_passes"] - s["baseline_passes"] for s in stats.values())
    if stats[guard]["baseline_passes"] == 0:
        return {
            **result,
            "status": "defer",
            "reason": "the fresh guard baseline did not reproduce",
        }
    unstable = [
        task
        for task, stat in stats.items()
        if stat["baseline_passes"] == stat["candidate_passes"] > stat["joint_successes"]
    ]
    if unstable:
        return {
            **result,
            "status": "defer",
            "reason": "opposite paired quality outcomes at equal pass counts",
            "repeat_tasks": unstable,
        }
    saved = math.fsum(s["joint_seconds_saved"] for s in stats.values())
    status = "advance" if gains > 0 or saved > 0 else "revise"
    return {
        **result,
        "status": status,
        "reason": "paired development comparison",
        "consequence": card["decisions"][status],
    }


def assigned_plan(card, report, history, enrollment):
    plan = decision(card, report, history)
    if plan["status"] == "run":
        slot = plan["next"]
        slot["assignment_id"] = digest(
            canonical([enrollment, len(history) + 1, slot]).encode()
        )
    return plan


def episode_ids(row):
    seen = set()
    for name, value in row["evidence"].items():
        path = Path(name)
        if path.name != "result.json":
            continue
        requests = row["evidence"].get(str(path.with_name("requests.jsonl")))
        if requests:
            seen.add(
                digest(
                    canonical(
                        {"result.json": value, "requests.jsonl": requests}
                    ).encode()
                )
            )
    return seen


def historical_episodes(ledger):
    """Pin known development executions so re-wrapping their evidence cannot enroll them."""
    seen = set()
    for row in ledger.rows():
        if row["kind"] == "measurement" and row["split"] == "dev":
            seen.update(episode_ids(row))
    return sorted(seen)


def replay_attempts(ledger, card, report, parent, state, attempt_file=None):
    """Recheck a snapshot and optional new observation without writing state."""
    history = state["attempts"]
    enrollment = sha(state["enrollment"])
    inputs = []
    for previous in history:
        ref = {
            k: previous["evidence"][k]
            for k in ("measurement", "invalid_run", "arm")
            if k in previous["evidence"]
        }
        inputs.append(({**previous, **ref}, previous))
    if attempt_file:
        new = parse(attempt_file.read_bytes())
        if "invalid_run" in new:
            new["invalid_run"] = str(
                (attempt_file.parent / new["invalid_run"]).resolve()
            )
        inputs.append((new, None))
    replay = []
    physical = {}
    for attempt, old in inputs:
        check(
            attempt["task"] in (card["target_task"], card["guard_task"]),
            "task outside development card",
        )
        item = collect(ledger, attempt, parent, report)
        identity = item["episode_sha256"]
        check(
            identity not in state["historical_episodes"],
            "physical episode predates this prospective enrollment",
        )
        if old is None and physical.get(identity) == item:
            continue  # idempotent re-delivery of exactly this observation
        check(identity not in physical, "physical episode reused")
        if old is not None:
            check(old == item, "stored attempt differs from original evidence")
        plan = assigned_plan(card, report, replay, enrollment)
        check(
            plan["status"] == "run"
            and all(item[k] == v for k, v in plan["next"].items()),
            "attempt does not match next scheduled slot",
        )
        physical[identity] = item
        replay.append(item)
    return replay


def schedule(ledger, args):
    if __package__:
        from .loop import writer_lock
    else:
        from loop import writer_lock
    card, card_hash, report, parent = checked_protocol(ledger, args.card)
    workspace = ledger.path.parent / "schedules" / card["study"]
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "state.json"
    with writer_lock(workspace / "writer.lock"):
        state = {"card_sha256": card_hash, "attempts": []}
        if path.exists():
            state = parse(path.read_bytes())
        check(
            state["card_sha256"] == card_hash,
            "experiment card changed after enrollment",
        )
        history = state["attempts"]
        enroll = "enrollment" not in state
        if enroll:
            check(
                not history,
                "legacy nonempty schedule lacks prospective assignments; retain as historical",
            )
            state["enrollment"] = secrets.token_hex(32)
            state["historical_episodes"] = historical_episodes(ledger)
        replay = replay_attempts(ledger, card, report, parent, state, args.attempt)
        if enroll or len(replay) != len(history):
            state["attempts"] = replay
            with tempfile.NamedTemporaryFile(dir=workspace, delete=False) as f:
                staging = Path(f.name)
            try:
                staging.write_text(canonical(state) + "\n", encoding="utf-8")
                os.replace(staging, path)
            finally:
                staging.unlink(missing_ok=True)
        plan = assigned_plan(card, report, replay, state["enrollment"])
        return {
            "study": card["study"],
            "card_sha256": card_hash,
            "state": str(path.resolve()),
            **plan,
        }
