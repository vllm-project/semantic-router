"""Read-only enrollment check for an equal-budget comparison of recipe search methods."""

from pathlib import Path

import yaml

if __package__:
    from .loop_evidence import check, component, digest, sha
    from .schedule import canonical, parse
    from .signature import prefix_receipts, task_signature
    from .target_dev import prompt_identities
    from .window_action import apply_window, load_space, window_bounds
else:
    from loop_evidence import check, component, digest, sha
    from schedule import canonical, parse
    from signature import prefix_receipts, task_signature
    from target_dev import prompt_identities
    from window_action import apply_window, load_space, window_bounds


METHODS = ["existing_skill", "random", "guided"]


def add_arguments(parser):
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--schedule-card",
        type=Path,
        action="append",
        default=[],
        help="completed development schedule releasing an assigned prefix",
    )
    parser.add_argument(
        "--target-prefixes",
        action="append",
        default=[],
        metavar="DEV_MEASUREMENT",
        help="verified prefix measurement from the allocated development tasks (repeatable)",
    )
    parser.add_argument(
        "--candidate-trials-per-method",
        type=int,
        help="number of distinct candidate configurations each method may try",
    )
    parser.add_argument(
        "--target-dev-manifest",
        type=Path,
        help="frozen development instructions and the solver's prompt template",
    )
    parser.add_argument(
        "--target-dev-manifest-sha256",
        help="manifest digest from the independent development packet export",
    )


def pinned_file(parent, name, fingerprint):
    check(
        isinstance(name, str)
        and Path(name).name == name
        and name not in {"", ".", ".."},
        "use a sibling metadata file",
    )
    path = (parent / name).resolve(strict=True)
    check(path.parent == parent.resolve(), "metadata file outside protocol directory")
    raw = path.read_bytes()
    check(digest(raw) == sha(fingerprint), "metadata fingerprint mismatch")
    return raw


def split_checks(protocol, folder):
    refs = protocol["split_plans"]
    check(
        isinstance(refs, dict) and set(refs) == {"dev", "holdout"},
        "declare both split plan slots",
    )
    identities, ids_by_split, blockers = {}, {}, []
    for split in ("dev", "holdout"):
        ref = refs[split]
        if ref is None:
            blockers.append(f"needs_independent_{split}_plan")
            identities[split] = None
            ids_by_split[split] = set()
            continue
        check(
            isinstance(ref, dict) and set(ref) == {"file", "sha256"},
            "invalid split plan reference",
        )
        plan = parse(pinned_file(folder, ref["file"], ref["sha256"]))
        check(
            isinstance(plan, dict)
            and set(plan)
            == {"schema", "study", "split", "methods", "instruction_sha256"},
            "expected split allocation metadata only",
        )
        check(
            plan["schema"] == "recipe-lab-search-split-v1"
            and plan["study"] == protocol["study"]
            and plan["split"] == split
            and plan["methods"] == METHODS,
            "incompatible split allocation",
        )
        ids = plan["instruction_sha256"]
        check(
            isinstance(ids, list)
            and ids
            and all(sha(i) for i in ids)
            and len(set(ids)) == len(ids),
            "invalid or duplicated task fingerprints",
        )
        check(
            not set(ids).intersection(ids_by_split.get("dev", set())),
            "holdout overlaps allocated development",
        )
        ids_by_split[split] = set(ids)
        identities[split] = {"sha256": ref["sha256"], "tasks": len(ids)}
    return identities, ids_by_split, blockers


def instruction_ids(row):
    ids = {
        name: sha(case["controls"]["case"]["instruction_sha256"])
        for name, case in row["cases"].items()
    }
    check(
        ids and len(set(ids.values())) == len(ids), "duplicate or missing task identity"
    )
    return ids


def confirm_task_manifest(row, declared, prompt_map, identity_binding):
    files = [Path(f) for f in row["evidence"] if Path(f).name == "frozen.json"]
    check(len(files) == 1, "expected one frozen task manifest")
    raw = files[0].read_bytes()
    check(digest(raw) == row["evidence"][str(files[0])], "frozen task manifest changed")
    packet = parse(raw)
    check(
        packet.get("split", "dev") == "dev", "frozen packet is not development evidence"
    )
    check(
        {name: item["instruction_sha256"] for name, item in packet["cases"].items()}
        == declared,
        "prefix tasks differ from frozen task manifest",
    )
    if prompt_map:
        check(
            packet.get("split") == "dev"
            and packet.get("solver_files", {}).get(
                identity_binding.get("template_file", "prompt-template.txt")
            )
            == identity_binding["template_sha256"],
            "solver template differs from frozen development measurement",
        )
        if identity_binding["method"] == "swe_solver_template":
            for name, observed in declared.items():
                check(
                    packet["cases"][name].get("repo") == prompt_map[observed]["repo"],
                    "repo differs from frozen development measurement",
                )


def preflight(ledger, args):
    if __package__:
        from .loop import (
            baseline_config,
            check_visible,
            comparable,
            released_assignments,
            same_deployment,
            verify,
        )
    else:
        from loop import (
            baseline_config,
            check_visible,
            comparable,
            released_assignments,
            same_deployment,
            verify,
        )

    raw = args.protocol.read_bytes()
    protocol = parse(raw)
    check(
        isinstance(protocol, dict)
        and set(protocol)
        == {
            "schema",
            "study",
            "based_on",
            "prefixes",
            "space",
            "space_sha256",
            "seed",
            "budget",
            "split_plans",
        },
        "invalid preflight protocol fields",
    )
    check(
        protocol["schema"] == "recipe-lab-search-preflight-v1",
        "unknown search protocol",
    )
    component(protocol["study"])
    seed, budget = protocol["seed"], protocol["budget"]
    trials = getattr(args, "candidate_trials_per_method", None)
    check(type(seed) is int and seed >= 0, "a nonnegative integer seed is required")
    check(
        trials is None or (type(trials) is int and trials > 0),
        "candidate trials per method must be a positive integer",
    )
    check(
        isinstance(budget, dict)
        and set(budget) == {"episodes", "proposal_model_tokens"}
        and all(type(n) is int and n > 0 for n in budget.values()),
        "declare a shared positive whole-episode and proposer-token budget",
    )
    base, prefixes = (
        ledger.measure(protocol["based_on"]),
        ledger.measure(protocol["prefixes"]),
    )
    check(
        base["split"] == prefixes["split"] == "dev",
        "preflight requires development measurements",
    )
    allocations, ids, plan_blockers = split_checks(protocol, args.protocol.parent)
    known = set(instruction_ids(base).values()) | set(
        instruction_ids(prefixes).values()
    )
    targets = [ledger.measure(ref) for ref in getattr(args, "target_prefixes", [])]
    check(
        not known.intersection(ids["holdout"]), "holdout overlaps observed development"
    )
    check(
        all(row["split"] == "dev" for row in targets),
        "target prefixes require development measurements",
    )
    manifest = getattr(args, "target_dev_manifest", None)
    manifest_sha = getattr(args, "target_dev_manifest_sha256", None)
    check(
        (manifest is None) == (manifest_sha is None) and (manifest is None or targets),
        "supply target prefixes and both development manifest arguments together",
    )
    prompt_map, identity_binding = {}, {"method": "same_instruction_sha256"}
    if manifest is not None:
        prompt_map, identity_binding = prompt_identities(
            manifest, manifest_sha, protocol["study"], ids["dev"]
        )
    if not targets:
        check(
            not known.intersection(ids["dev"]),
            "split overlaps observed development; supply target prefixes to check the allocated dev tasks",
        )
    target_tasks = {}
    for row in targets:
        check(
            row["config_sha256"] == base["config_sha256"],
            "target prefixes must originate from the frozen parent configuration",
        )
        names = instruction_ids(row)
        if prompt_map:
            check(
                set(names.values()) <= prompt_map.keys(),
                "solver prompt outside frozen development manifest",
            )
        mapped = {
            name: prompt_map[fp]["raw"] if prompt_map else fp
            for name, fp in names.items()
        }
        claimed = set(mapped.values())
        check(
            claimed <= ids["dev"],
            "target prefixes must belong to the frozen development allocation",
        )
        check(
            not target_tasks.keys() & claimed,
            "duplicate target development prefix task",
        )
        target_tasks.update(
            {
                identity: {
                    "task": name,
                    "measurement": row["record_id"],
                    "observed_instruction_sha256": names[name],
                }
                for name, identity in mapped.items()
            }
        )
    # Check all declared task identities before opening any episode or request logs.
    check_visible(
        [base, prefixes, *targets],
        released_assignments(ledger, getattr(args, "schedule_card", [])),
    )
    for row in targets:
        confirm_task_manifest(row, instruction_ids(row), prompt_map, identity_binding)
    verify(base)
    if prefixes["record_id"] != base["record_id"]:
        verify(prefixes)
    comparable(base, prefixes)
    doc = baseline_config(base)
    space = load_space(
        pinned_file(args.protocol.parent, protocol["space"], protocol["space_sha256"])
    )
    check(
        space["public_model"] == base["public_model"]
        and space["base_config_sha256"] == base["config_sha256"],
        "space differs from the predeclared parent",
    )
    historical = action_report(base, doc, space, list(prefix_receipts(prefixes)), seed)
    active = historical
    comparison_source = "historical_prefixes"
    target_report = {"status": "not_checked"}
    blockers = list(plan_blockers)
    target_observed = False
    if targets:
        traces = []
        for row in targets:
            verify(row)
            same_deployment(base, row)
            names = instruction_ids(row)
            traces.extend(
                (prompt_map[names[t]]["raw"] if prompt_map else names[t], data, fp)
                for t, data, fp in prefix_receipts(row)
            )
        active = action_report(base, doc, space, traces, seed)
        comparison_source = "target_dev_prefixes"
        missing = ids["dev"] - target_tasks.keys()
        if missing:
            blockers.append("missing_target_development_tasks")
        unobserved = [
            t for t, c in active["coverage_by_task"].items() if not c["observed"]
        ]
        if unobserved:
            blockers.append("unobservable_target_development_task")
        target_observed = not missing and not unobserved
        target_report = {
            "status": "checked",
            "tasks_by_instruction": target_tasks,
            "identity_binding": identity_binding,
            "missing_instruction_sha256": sorted(missing),
            **active,
        }
    if active["distinct_new_classes"] < 2:
        blockers.insert(0, "insufficient_distinguishable_new_actions")
    remaining_classes = None
    selection_headroom = "unknown"
    if target_observed and trials is not None:
        remaining_classes = max(0, active["distinct_new_classes"] - trials)
        selection_headroom = "non_exhaustive" if remaining_classes else "exhaustive"
        if not remaining_classes:
            blockers.append("candidate_trials_cover_all_distinguishable_new_actions")
    status = "target_actions_distinguished" if targets else "protocol_ready"
    return {
        "schema": "recipe-lab-search-preflight-report-v2",
        "study": protocol["study"],
        "protocol_sha256": digest(raw),
        "status": "insufficient" if blockers else status,
        "blockers": blockers,
        "based_on": base["record_id"],
        "prefixes": prefixes["record_id"],
        "space_sha256": protocol["space_sha256"],
        "allowed_upper": space["allowed_upper"],
        **{
            k: v
            for k, v in historical.items()
            if k not in {"shared_class_representatives", "random_class_order"}
        },
        "target_development": target_report,
        "comparison": {
            "methods": METHODS,
            "shared_budget_per_method": budget,
            "split_plans": allocations,
            "class_representative_source": comparison_source,
            "candidate_trials_per_method": trials,
            "target_selection_headroom": selection_headroom,
            "target_new_classes_beyond_trials": remaining_classes,
            "shared_class_representatives": active["shared_class_representatives"],
            "random_seed": seed,
            "random_class_order": active["random_class_order"],
            "sampling_rule": "order one representative per changed observed class by SHA-256 of [seed, upper]",
        },
        "native_grader_qualification": "not_assessed",
        "paid_search_authorized": None,
        "search_executed": False,
        "scope": "Fixed-prefix predicate coverage. The top-level signatures describe the protocol's historical prefix record. Target signatures use only supplied records matched to the frozen dev allocation. Whole-task performance, grader qualification and search execution require separate evidence.",
    }


def action_report(base, doc, space, traces, seed):
    lower, current = window_bounds(doc, space)
    hashes, coverage, actions, classes = {}, {}, [], {}
    # The current setting always participates as a reference, even when its
    # no-op value was deliberately excluded from the declared edit domain.
    for upper in sorted({current, *space["allowed_upper"]}):
        cfg_hash = base["config_sha256"]
        if upper != current:
            candidate, _ = apply_window(
                doc,
                space,
                public_model=base["public_model"],
                config_sha256=cfg_hash,
                upper=upper,
            )
            cfg_hash = digest(
                yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True).encode()
            )
        hits = {}
        for task, receipt, fingerprint in traces:
            row = task_signature(
                receipt,
                base["public_model"],
                {"lower": lower, "before_upper": current, "after_upper": upper},
            )
            hashes[task] = fingerprint
            coverage[task] = {
                k: row[k] for k in ("observed", "unknown", "missing_by_reason")
            }
            hits[task] = row["after_hit_calls"]
        sig = digest(canonical(hits).encode())
        actions.append(
            {
                "upper": upper,
                "config_sha256": cfg_hash,
                "observed_predicate_sha256": sig,
                "parent": upper == current,
            }
        )
        classes.setdefault(sig, []).append(upper)
    candidates = [min(bounds) for bounds in classes.values() if current not in bounds]
    permutation = sorted(
        candidates, key=lambda upper: digest(canonical([seed, upper]).encode())
    )
    return {
        "actions": actions,
        "observed_equivalence_classes": classes,
        "distinct_new_classes": len(candidates),
        "coverage_by_task": coverage,
        "prefix_receipt_sha256": hashes,
        "shared_class_representatives": sorted(candidates),
        "random_class_order": permutation,
    }
