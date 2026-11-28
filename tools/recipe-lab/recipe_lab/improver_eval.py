"""Freeze three development selections and evaluate one independent, audited matrix."""

import math
import secrets
from pathlib import Path

if __package__:
    from . import decision_summary, search_accounting, search_protocol
    from .headroom import paired, read_matrix, summarize
    from .loop_evidence import check, component, digest, sha
    from .schedule import canonical, parse
else:
    import decision_summary
    import search_accounting
    import search_protocol
    from headroom import paired, read_matrix, summarize
    from loop_evidence import check, component, digest, sha
    from schedule import canonical, parse


METHODS = search_protocol.METHODS
CONTROLS = ["baseline", "direct_strong"]


def add_arguments(subs):
    freeze = subs.add_parser(
        "search-freeze",
        help="freeze three development selections before independent grading",
    )
    search_protocol.add_arguments(freeze)
    freeze.add_argument("--selections", required=True, type=Path)
    evaluate = subs.add_parser(
        "improver-eval",
        help="score a frozen three-method study from audited holdout receipts",
    )
    evaluate.add_argument("--study", required=True)
    evaluate.add_argument("--manifest", required=True, type=Path)


def pinned_path(folder, ref):
    check(
        isinstance(ref, dict) and set(ref) == {"file", "sha256"},
        "expected pinned sibling file",
    )
    raw = search_protocol.pinned_file(folder, ref["file"], ref["sha256"])
    return folder / ref["file"], raw


def read_method(ledger, folder, method, spec, preflight, parent, used, enrollment):
    check(
        set(spec) == {"policy_sha256", "trial_cards", "usage"},
        "invalid method selection",
    )
    policy = sha(spec["policy_sha256"])
    cards = spec["trial_cards"]
    meta = preflight["comparison"]
    check(
        isinstance(cards, list) and len(cards) <= meta["candidate_trials_per_method"],
        "trial ceiling exceeded",
    )
    target = preflight["target_development"]
    eligible = {a["upper"]: a["config_sha256"] for a in target["actions"]}
    common = meta["shared_class_representatives"]
    configs = {eligible[u] for u in common}
    allocation = {
        r["task"]: ledger.measure(r["measurement"])["cases"][r["task"]]["controls"]
        for r in target["tasks_by_instruction"].values()
    }
    allowed_selection = {parent}
    trials, evidence = [], []
    spend, invalid, valid_seconds, invalid_seconds = 0, 0, 0.0, 0.0
    for slot, ref in enumerate(cards):
        path, card_bytes = pinned_path(folder, ref)
        outcome = decision_summary.summarize(ledger, path)
        check(
            outcome["complete"]
            and outcome["card_sha256"] == digest(card_bytes)
            and outcome["parent_config_sha256"] == parent,
            "unclosed or different-parent development trial",
        )
        for task, control_sha in outcome["task_controls_sha256"].items():
            check(
                task in allocation
                and control_sha == digest(canonical(allocation[task]).encode()),
                "trial differs from frozen target task or development controls",
            )
        candidate = outcome["candidate_sha256"]
        check(
            candidate in configs and candidate not in trials,
            "trial outside shared representatives or repeated",
        )
        check(outcome["state_sha256"] not in used, "trial state reused across methods")
        used.add(sha(outcome["state_sha256"]))
        if method == "random":
            check(
                candidate == eligible[meta["random_class_order"][slot]],
                "trial differs from frozen random order",
            )
        for task, group in outcome["by_task"].items():
            for attempt in group["attempts"]:
                key = sha(attempt["episode_sha256"])
                check(
                    key not in used, "development episode reused across search trials"
                )
                used.add(key)
                if attempt["valid"]:
                    row = ledger.measure(attempt["measurement"])
                    check(
                        row["cases"][task]["controls"] == allocation.get(task),
                        "development trial controls differ",
                    )
            # Also bind planned tasks when all their attempts were invalid or a trial stopped early.
            check(task in allocation, "unallocated trial task")
        costs = outcome["cost"]
        spend += costs["spent_episodes"]
        invalid += costs["invalid_attempts"]
        valid_seconds += costs["valid_agent_seconds"]
        invalid_seconds += costs["invalid_attempt_wall_seconds"]
        if outcome["decision"] == "advance":
            allowed_selection.add(candidate)
        trials.append(candidate)
        card = parse(card_bytes)
        evidence.append(
            {
                "study": outcome["study"],
                "decision": outcome["decision"],
                "card_sha256": outcome["card_sha256"],
                "state_sha256": sha(outcome["state_sha256"]),
                "guard": outcome["guard_task"],
                "target": outcome["target_task"],
                "repeats": card["repeats"],
                "episode_ceiling": card["budget_episodes"],
            }
        )
    check(policy in allowed_selection, "selection lacks an advancing development trial")
    slots = search_accounting.check_trial_slots(
        ledger, enrollment, method, [t["study"] for t in evidence]
    )
    tokens = search_accounting.proposal_usage(
        folder, spec["usage"], used, enrollment, method
    )
    budget = meta["shared_budget_per_method"]
    check(
        spend <= budget["episodes"] and tokens <= budget["proposal_model_tokens"],
        "method exceeds its budget",
    )
    return {
        "policy_sha256": policy,
        "tried": trials,
        "trial_evidence": evidence,
        "slots": slots,
        "dev_episodes": spend,
        "invalid_dev_episodes": invalid,
        "valid_agent_seconds": valid_seconds,
        "invalid_attempt_wall_seconds": invalid_seconds,
        "reported_proposal_tokens": tokens,
    }


def freeze(ledger, args):
    report = search_protocol.preflight(ledger, args)
    check(
        report["status"] == "target_actions_distinguished"
        and report["comparison"]["target_selection_headroom"] == "non_exhaustive",
        "a ready, non-exhaustive target development preflight is required",
    )
    raw = args.selections.read_bytes()
    spec = parse(raw)
    check(
        set(spec)
        == {
            "schema",
            "protocol_sha256",
            "methods",
            "direct_strong_sha256",
            "holdout_repeats",
        }
        and spec["schema"] == "recipe-lab-search-selections-v1"
        and spec["protocol_sha256"] == report["protocol_sha256"],
        "selections differ from protocol",
    )
    check(set(spec["methods"]) == set(METHODS), "declare all three methods")
    n = spec["holdout_repeats"]
    check(type(n) is int and n > 0, "positive holdout repeat count required")
    protocol_bytes = args.protocol.read_bytes()
    check(
        digest(protocol_bytes) == report["protocol_sha256"], "search protocol changed"
    )
    _, allocated, blockers = search_protocol.split_checks(
        parse(protocol_bytes), args.protocol.parent
    )
    check(not blockers, "incomplete split allocations")
    parent = ledger.measure(report["based_on"])["config_sha256"]
    enrollment = search_accounting.load_enrollment(ledger, report)
    used = set()
    selected = {
        m: read_method(
            ledger,
            args.selections.parent,
            m,
            spec["methods"][m],
            report,
            parent,
            used,
            enrollment,
        )
        for m in METHODS
    }
    stages = {
        (t["guard"], t["target"], t["repeats"], t["episode_ceiling"])
        for method in selected.values()
        for t in method["trial_evidence"]
    }
    check(len(stages) <= 1, "methods must use the same paired development stage")
    identity = {
        "schema": "recipe-lab-search-freeze-v1",
        "study": report["study"],
        "protocol_sha256": report["protocol_sha256"],
        "selections_sha256": digest(raw),
        "search_enrollment": enrollment["enrollment"],
        "preflight_sha256": digest(canonical(report).encode()),
        "budget": report["comparison"]["shared_budget_per_method"],
        "methods": selected,
        "controls": {
            "baseline": parent,
            "direct_strong": sha(spec["direct_strong_sha256"]),
        },
        "holdout_instructions": sorted(allocated["holdout"]),
        "holdout_repeats": n,
    }
    from_loop = loop_io()
    folder = ledger.path.parent / "search-studies" / component(report["study"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "freeze.json"
    # Slot closure and the one-time claim of holdout instructions share the enrollment lock.
    with from_loop.writer_lock(
        search_accounting.enrollment_root(ledger) / "registry.lock"
    ):
        for method, result in selected.items():
            listed = [trial["study"] for trial in result["trial_evidence"]]
            check(
                result["slots"]
                == search_accounting.check_trial_slots(
                    ledger, enrollment, method, listed
                ),
                "trial slots changed while freezing the selection",
            )
        if path.exists():
            stored = parse(path.read_bytes())
            check(
                {k: v for k, v in stored.items() if k != "enrollment"} == identity,
                "study is already frozen",
            )
        else:
            seen = {
                i
                for row in ledger.rows()
                if row["kind"] == "measurement" and row["split"] == "holdout"
                for i in search_protocol.instruction_ids(row).values()
            }
            for previous in folder.parent.glob("*/freeze.json"):
                seen.update(parse(previous.read_bytes())["holdout_instructions"])
            check(
                not seen.intersection(allocated["holdout"]),
                "allocated holdout already recorded",
            )
            stored = {**identity, "enrollment": secrets.token_hex(32)}
            from_loop.publish_config(path, canonical(stored) + "\n")
        fingerprint = digest(path.read_bytes())
        return {
            "study": report["study"],
            "freeze_sha256": fingerprint,
            "freeze": str(path),
            "methods": selected,
            "budget": identity["budget"],
            "holdout_assignments": holdout_assignments(
                fingerprint, identity["holdout_instructions"], n
            ),
        }


def loop_io():
    if __package__:
        from . import loop
    else:
        import loop
    return loop


def assignment_id(fingerprint, arm, instruction, repeat):
    return digest(canonical([fingerprint, arm, instruction, repeat]).encode())


def holdout_assignments(fingerprint, instructions, repeats):
    arms = CONTROLS + METHODS
    assignments = []
    for rep in range(1, repeats + 1):
        for i, instruction in enumerate(instructions):
            start = (rep - 1 + i) % len(arms)
            for arm in arms[start:] + arms[:start]:
                assignments.append(
                    {
                        "arm": arm,
                        "instruction_sha256": instruction,
                        "repeat": rep,
                        "assignment_id": assignment_id(
                            fingerprint, arm, instruction, rep
                        ),
                    }
                )
    return assignments


def verify_holdout_sources(folder, row, fingerprint, valid):
    expected = assignment_id(
        fingerprint, row["arm"], row["instruction_sha256"], row["repeat"]
    )
    _, binding = pinned_path(folder, row["binding"])
    _, result = pinned_path(folder, row["outcome"])
    binding, result = parse(binding), parse(result)
    check(
        binding["assignment_id"] == result["assignment_id"] == expected
        and all(
            binding[k] == row[k]
            for k in ("instruction_sha256", "policy_sha256", "controls_sha256")
        ),
        "holdout assignment does not match adapter binding",
    )
    check(
        result["episode_sha256"] == row["episode_sha256"]
        and type(result["seconds"]) in (int, float)
        and result["seconds"] == row["seconds"],
        "holdout source time or episode differs",
    )
    if valid:
        check(
            result["assessment"] == "valid" and result["strict"] is row["strict"],
            "adapter strict grade differs",
        )
    else:
        check(
            result["assessment"] == "invalid" and result["reason"] == row["reason"],
            "adapter invalid outcome differs",
        )


def method_comparison(fixed, policies, baseline, candidate):
    if policies[baseline] == policies[candidate]:
        return {
            "same_policy": True,
            "scope": "Both methods delivered the same policy. Compare search costs.",
        }
    return {"same_policy": False, **paired(fixed[baseline], fixed[candidate])}


def score(study, fingerprint, manifest_path):
    manifest, manifest_sha, fixed = read_matrix(
        manifest_path, "holdout", sealed_search=True
    )
    policies = {
        **study["controls"],
        **{m: study["methods"][m]["policy_sha256"] for m in METHODS},
    }
    check(
        manifest["study"] == study["study"]
        and manifest["freeze_sha256"] == fingerprint
        and set(fixed) == set(policies),
        "matrix differs from frozen study or arms",
    )
    check(
        set(manifest["repeats"]) == set(range(1, study["holdout_repeats"] + 1)),
        "holdout repeats changed",
    )
    tasks, used, controls = {}, set(), {}
    for arm, cells in fixed.items():
        for (task, _), row in cells.items():
            identity = sha(row["instruction_sha256"])
            check(
                tasks.setdefault(task, identity) == identity,
                "holdout instruction changed",
            )
            check(
                row["freeze_sha256"] == fingerprint
                and row["policy_sha256"] == policies[arm],
                "episode lacks the frozen assignment or uses a different policy",
            )
            verify_holdout_sources(manifest_path.parent, row, fingerprint, True)
            used.add(row["episode_sha256"])
            controls[task] = row["controls_sha256"]
    check(
        sorted(tasks.values()) == study["holdout_instructions"],
        "holdout allocation differs from freeze",
    )
    invalid = manifest["invalid_attempts"]
    check(
        isinstance(invalid, list),
        "declare invalid holdout attempts, including an empty list",
    )
    invalid_cost = {arm: {"attempts": 0, "wall_seconds": 0.0} for arm in policies}
    for row in invalid:
        arm, task, elapsed = row["arm"], row["task"], row["seconds"]
        check(
            arm in fixed
            and task in tasks
            and type(row["repeat"]) is int
            and row["repeat"] in manifest["repeats"]
            and row["freeze_sha256"] == fingerprint
            and row["policy_sha256"] == policies[arm]
            and row["controls_sha256"] == controls[task]
            and row["instruction_sha256"] == tasks[task]
            and isinstance(row["reason"], str)
            and row["reason"].strip(),
            "unbound invalid holdout attempt",
        )
        key = sha(row["episode_sha256"])
        check(
            key not in used
            and type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and elapsed >= 0,
            "invalid or reused holdout attempt",
        )
        verify_holdout_sources(manifest_path.parent, row, fingerprint, False)
        used.add(key)
        invalid_cost[arm]["attempts"] += 1
        invalid_cost[arm]["wall_seconds"] += elapsed
    by_task = {
        task: {
            arm: [
                {
                    "repeat": rep,
                    "strict": fixed[arm][task, rep]["strict"],
                    "seconds": fixed[arm][task, rep]["seconds"],
                    "episode_sha256": fixed[arm][task, rep]["episode_sha256"],
                }
                for rep in manifest["repeats"]
            ]
            for arm in policies
        }
        for task in tasks
    }
    return {
        "schema": "recipe-lab-improver-eval-v1",
        "study": study["study"],
        "freeze_sha256": fingerprint,
        "manifest_sha256": manifest_sha,
        "scores_sha256": manifest["scores_sha256"],
        "methods": study["methods"],
        "shared_budget": study["budget"],
        "holdout": {arm: summarize(cells.values()) for arm, cells in fixed.items()},
        "by_task": by_task,
        "invalid_holdout": invalid_cost,
        "versus_controls": {
            m: {c: method_comparison(fixed, policies, c, m) for c in CONTROLS}
            for m in METHODS
        },
        "guided_versus_other_methods": {
            m: method_comparison(fixed, policies, m, "guided")
            for m in METHODS
            if m != "guided"
        },
        "scope": "Descriptive paired outcomes for one study. Adapter-audited costs and strict grades. Generalization requires a new project.",
    }


def evaluate(ledger, args):
    folder = ledger.path.parent / "search-studies" / component(args.study)
    path = folder / "freeze.json"
    raw = path.read_bytes()
    study, fingerprint = parse(raw), digest(raw)
    check(
        study["study"] == args.study and sha(study["enrollment"]), "wrong search freeze"
    )
    result_path = folder / "evaluation.json"
    api = loop_io()
    with api.writer_lock(folder / "study.lock"):
        if result_path.exists():
            existing = parse(result_path.read_bytes())
            check(
                existing["freeze_sha256"] == fingerprint
                and existing["manifest_sha256"] == digest(args.manifest.read_bytes()),
                "holdout already evaluated",
            )
        report = score(study, fingerprint, args.manifest)
        if result_path.exists():
            check(existing == report, "evaluation differs from sealed result")
        else:
            api.publish_config(result_path, canonical(report) + "\n")
        return report
