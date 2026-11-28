"""Plan the next development probe on one deployed recipe and its recorded traffic."""

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml

if __package__:
    from .loop_evidence import check, digest, fit_status, require_isolated, target
    from .planning_replay import replay_task
    from .planning_space import (
        build_space,
        extract_candidate,
        extract_reference,
        render_handoff,
    )
    from .semantic_depth import add_candidate
    from .signature import prefix_receipts
else:
    from loop_evidence import check, digest, fit_status, require_isolated, target
    from planning_replay import replay_task
    from planning_space import (
        build_space,
        extract_candidate,
        extract_reference,
        render_handoff,
    )
    from semantic_depth import add_candidate
    from signature import prefix_receipts


def add_arguments(subs):
    inventory = subs.add_parser(
        "plan-space", help="inventory a deployed recipe and candidate edits"
    )
    inventory.add_argument("--config", required=True, type=Path)
    inventory.add_argument("--public-model", required=True)
    inventory.add_argument("--strong", required=True)
    dev = subs.add_parser(
        "plan-dev", help="rank candidate probes using verified development requests"
    )
    dev.add_argument("--base", required=True, help="development parent record ID")
    dev.add_argument(
        "--depth-evidence",
        type=Path,
        help="development SystemOne scores, competing branches, and eligibility bound to parent calls",
    )
    dev.add_argument(
        "--prefixes",
        action="append",
        default=[],
        metavar="ID",
        help="additional measurements of exactly the parent config; repeatable",
    )
    dev.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="ID",
        help="paired dev comparison to use as feedback; repeatable",
    )
    dev.add_argument(
        "--schedule-card",
        type=Path,
        action="append",
        default=[],
        help="completed paired study authorizing its assigned measurements",
    )
    dev.add_argument(
        "--out-candidate",
        type=Path,
        help="publish the recommended candidate for probing",
    )
    dev.add_argument(
        "--probe-receipt",
        type=Path,
        help="review fixture or real public probe evidence for this draft",
    )
    dev.add_argument(
        "--probe-config",
        type=Path,
        help="deployed config when isolated endpoints change the source hash",
    )
    for parser in (inventory, dev):
        parser.add_argument(
            "--fast",
            required=True,
            help="fast provider model binding in this deployment",
        )
        parser.add_argument(
            "--reference",
            type=Path,
            action="append",
            default=[],
            help="reference config, repeatable; uses matching or sole public entry",
        )
        parser.add_argument(
            "--reference-entry",
            nargs=2,
            action="append",
            default=[],
            metavar=("PUBLIC", "CONFIG"),
            help="select one public entry in a reference",
        )
        anchor = parser.add_mutually_exclusive_group()
        anchor.add_argument(
            "--anchor",
            type=Path,
            help="reference mechanism to reproduce before exploring edits",
        )
        anchor.add_argument(
            "--anchor-entry",
            nargs=2,
            metavar=("PUBLIC", "CONFIG"),
            help="select the nominated reference from a multi-entry config",
        )
        parser.add_argument(
            "--anchor-models",
            nargs=2,
            metavar=("SOURCE_STRONG", "SOURCE_FAST"),
            help="provider names for the two roles in the nominated reference",
        )
        parser.add_argument(
            "--reference-library",
            type=Path,
            help="inspect maintained recipe directories containing config.yaml",
        )
        parser.add_argument(
            "--edit-space",
            type=Path,
            action="append",
            default=[],
            help="enumerate allowed values of a pinned native edit space",
        )


def model_cards(doc):
    cards = doc["providers"]["models"]
    check(
        isinstance(cards, list)
        and cards
        and all(
            isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"]
            for m in cards
        ),
        "expected a nonempty providers.models list of named model cards",
    )
    by_name = {m["name"]: m for m in cards}
    check(len(by_name) == len(cards), "ambiguous provider model names")
    return by_name


def upstream(card):
    return card.get("provider_model_id") or card["name"]


def reference_candidate(
    doc, explicit, *, public, cards, strong, fast, source_models=None
):
    names = [
        name for entry in doc["entrypoints"] for name in entry.get("model_names", [])
    ]
    selected = explicit
    if not selected:
        if public in names:
            selected = public
        elif len(names) == 1:
            selected = names[0]
    if explicit:
        check(explicit in names, f"reference public entry not found: {explicit}")
    selected_cards = model_cards(doc) if "providers" in doc else {}
    if source_models:
        check(
            len(set(source_models)) == 2
            and all(n in selected_cards for n in source_models),
            "anchor models must be distinct provider names from the reference",
        )
    if not selected:
        return {
            "kind": "requires_probe",
            "predicate": None,
            "reason": "select_reference_entry",
            "public_entries": names,
        }
    recipe = target(doc, selected)
    decisions = recipe.get("decisions", [])
    if not selected_cards:
        return {
            "kind": "requires_probe",
            "predicate": None,
            "reason": "bind_reference_template_models",
            "public_entries": names,
            "public_model": selected,
            "reference_recipe": recipe["name"],
            "reference_decisions": [d["name"] for d in decisions],
            "reference_bindings": {
                d["name"]: {
                    "models": [ref["model"] for ref in d.get("modelRefs", [])],
                    "minimum_candidates": d.get("algorithm", {}).get(
                        "minimum_candidates"
                    ),
                }
                for d in decisions
            },
        }
    mapping = {}
    if source_models:
        mapping = dict(zip((strong, fast), source_models, strict=True))
    for model in (strong, fast):
        if model in mapping:
            continue
        matches = [
            name
            for name, card in selected_cards.items()
            if upstream(card) == upstream(cards[model])
        ]
        if len(matches) != 1:
            return {
                "kind": "requires_probe",
                "predicate": None,
                "reason": "reference_model_mapping_required",
                "public_model": selected,
                "reference_decisions": [d["name"] for d in decisions],
                "available_models": sorted(selected_cards),
            }
        mapping[model] = matches[0]
    candidate = extract_reference(
        doc, public_model=selected, strong=mapping[strong], fast=mapping[fast]
    )
    return {
        **candidate,
        "public_model": selected,
        "binding_map": mapping,
        "reference_fast_limits": {
            k: selected_cards[mapping[fast]].get(k)
            for k in ("context_window_size", "max_output_tokens")
        },
    }


def inventory_for(doc, source_sha, public, strong, args):
    fast = args.fast
    space = build_space(doc, public_model=public, strong=strong, fast=fast)
    cards = model_cards(doc)
    recipe = require_isolated(doc, public)
    candidates = list(space["candidates"])
    configs = {}
    if args.edit_space:
        if __package__:
            from .planning_space import native_edits
        else:
            from planning_space import native_edits
        used = set()
        for path in args.edit_space:
            raw_space = path.read_bytes()
            space_sha = digest(raw_space)
            check(space_sha not in used, "duplicate edit space")
            used.add(space_sha)
            for item, draft in native_edits(
                doc,
                raw_space,
                base_sha=source_sha,
                public_model=public,
                strong=strong,
                fast=fast,
            ):
                item["space_file"] = path.name
                configs[item["id"]] = yaml.safe_dump(
                    draft, sort_keys=False, allow_unicode=True
                ).encode("utf-8")
                candidates.append(item)
    refs = [(None, path) for path in args.reference]
    refs.extend((entry, Path(path)) for entry, path in args.reference_entry)
    if args.reference_library:
        library = sorted(args.reference_library.glob("*/config.yaml"))
        check(library, "reference library contains no */config.yaml files")
        refs.extend((None, path) for path in library)
    anchor = None
    source_models = getattr(args, "anchor_models", None)
    check(
        not source_models or args.anchor or args.anchor_entry,
        "--anchor-models requires a nominated anchor",
    )
    if args.anchor or args.anchor_entry:
        anchor = (
            (None, args.anchor)
            if args.anchor
            else (args.anchor_entry[0], Path(args.anchor_entry[1]))
        )
        refs.insert(0, anchor)
    seen = set()
    for explicit, path in refs:
        raw = path.read_bytes()
        identity = digest(raw)
        ref = yaml.safe_load(raw)
        check(
            isinstance(ref, dict) and ref.get("version") == "v0.5",
            f"expected v0.5 reference: {path.name}",
        )
        is_anchor = (explicit, path) == anchor
        item = reference_candidate(
            ref,
            explicit,
            public=public,
            cards=cards,
            strong=strong,
            fast=fast,
            source_models=source_models if is_anchor else None,
        )
        entry = item.get("public_model", explicit)
        if (identity, entry) in seen:
            continue
        seen.add((identity, entry))
        source = (
            str(Path(path.parent.name) / path.name)
            if path.name == "config.yaml"
            else path.name
        )
        item.update(
            id=f"reference-{len(seen)}",
            source=source,
            reference_sha256=identity,
            anchor=is_anchor,
        )
        # A config with different providers or a different public scope supplies only a mechanism.
        if entry == public and item["kind"] == "count_handoff":
            child = target(ref, public)
            restored = {
                **ref,
                "recipes": [recipe if r is child else r for r in ref["recipes"]],
            }
            routing = extract_candidate(
                ref, public_model=public, strong=strong, fast=fast
            )
            if (
                restored == doc
                and routing["kind"] == "count_handoff"
                and preserves_parent(recipe, child)
            ):
                configs[item["id"]] = raw
        candidates.append(item)
    for item in candidates:
        key = item["id"]
        predicate = item.get("predicate") or {}
        if (
            key not in configs
            and item["kind"] == "count_handoff"
            and predicate.get("feature") == "tool_results"
            and predicate.get("upper") is None
        ):
            try:
                draft = render_handoff(
                    doc,
                    public_model=public,
                    strong=strong,
                    fast=fast,
                    threshold=predicate["lower"],
                )
            except ValueError as exc:
                item["materialization_blocker"] = str(exc)
            else:
                configs[key] = yaml.safe_dump(
                    draft, sort_keys=False, allow_unicode=True
                ).encode("utf-8")
        item["candidate_sha256"] = digest(configs[key]) if key in configs else None
        if key in configs:
            concrete = yaml.safe_load(configs[key])
            effective = extract_candidate(
                concrete, public_model=public, strong=strong, fast=fast
            )
            if effective["kind"] == "count_handoff":
                actual = target(concrete, public)
                fallback = next(
                    d["name"] for d in actual["decisions"] if not d.get("rules")
                )
                item["public_probe_predicate"] = {
                    **effective["predicate"],
                    "decision": effective["decision"],
                    "fallback_decision": fallback,
                }
    input_config = doc.get("global", {}).get("router", {}).get("routing_input", {})
    return {
        **space,
        "schema": "recipe-lab-planning-space-v1",
        "public_model": public,
        "filters_user_messages": bool(input_config.get("skip_user_messages_matching")),
        "base_config_sha256": source_sha,
        "strong": strong,
        "fast": fast,
        "candidates": candidates,
    }, configs


def preserves_parent(parent, child):
    """Allow an existing reference file only for a pure count branch added to a single fallback."""
    routing_keys = {"description", "decisions", "signals"}
    if {k: v for k, v in parent.items() if k not in routing_keys} != {
        k: v for k, v in child.items() if k not in routing_keys
    }:
        return False
    lanes = parent.get("decisions", [])
    if (
        len(lanes) != 1
        or len(child["decisions"]) != 2
        or lanes[0] not in child["decisions"]
        or parent.get("signals")
    ):
        return False
    branch = next((d for d in child["decisions"] if d != lanes[0]), None)
    if not branch or set(child.get("signals", {})) != {"conversation"}:
        return False

    def names(node):
        if "name" in node:
            return {node["name"]}
        return {name for item in node.get("conditions", []) for name in names(item)}

    signals = child["signals"]["conversation"]
    if len(signals) != len(names(branch.get("rules") or {})) or {
        s["name"] for s in signals
    } != names(branch.get("rules") or {}):
        return False
    change = {"name", "description", "priority", "modelRefs", "rules"}
    return bool(
        {k: v for k, v in branch.items() if k not in change}
        == {k: v for k, v in lanes[0].items() if k not in change}
        and all(set(r) == {"model"} for r in branch["modelRefs"])
    )


def pending_assignment(row, released):
    for name, fingerprint in row["evidence"].items():
        path = Path(name)
        if path.parent.parent.name == "bindings" and path.suffix == ".json":
            raw = path.read_bytes()
            check(digest(raw) == fingerprint, "development binding changed")
            assignment = json.loads(raw).get("assignment_id")
            if assignment and (row["record_id"], assignment) not in released:
                return True
    return False


def independent_dev_row(row, released):
    if __package__:
        from .loop import verify
    else:
        from loop import verify
    check(row["split"] == "dev", "planning requires development measurements")
    check(
        not pending_assignment(row, released),
        "assigned episodes require a completed --schedule-card",
    )
    verify(row)


def observed_tasks(ledger, base, extra, released):
    ids = [base["record_id"], *extra]
    check(len(ids) == len(set(ids)), "duplicate prefix measurement")
    tasks, seen, controls = [], set(), {}
    for ident in ids:
        row = base if ident == base["record_id"] else ledger.measure(ident)
        check(
            row["config_sha256"] == base["config_sha256"]
            and row["public_model"] == base["public_model"]
            and row["strong_model"] == base["strong_model"],
            "prefixes must use the same parent configuration",
        )
        independent_dev_row(row, released)
        for task, raw, sha in prefix_receipts(row):
            case = row["cases"][task]
            check(
                controls.setdefault(task, case["controls"]) == case["controls"],
                f"task or evaluation protocol changed across prefixes for {task}",
            )
            check((task, sha) not in seen, "duplicate development trace")
            seen.add((task, sha))
            tasks.append(
                (
                    {
                        "task": task,
                        "measurement": ident,
                        "request_sha256": sha,
                        "instruction_sha256": case["controls"]["case"][
                            "instruction_sha256"
                        ],
                        "control_sha256": control_hash(case),
                        "parent_strict": case["correct"],
                        "parent_seconds": case["seconds"],
                        "parent_termination": case["termination"],
                    },
                    raw,
                )
            )
    return tasks


def control_hash(case):
    return digest(
        json.dumps(case["controls"], sort_keys=True, allow_nan=False).encode()
    )


def prior_results(ledger, ids, base, candidate_hashes, released, current_controls):
    if __package__:
        from .loop import comparable
    else:
        from loop import comparable
    feedback = []
    rows = ledger.rows()
    comparisons = {
        r["record_id"]: r
        for r in rows
        if r["kind"] == "comparison" and r["split"] == "dev"
    }
    measurements = {
        r["record_id"]: r
        for r in rows
        if r["kind"] == "measurement" and r["split"] == "dev"
    }
    selected = list(dict.fromkeys(ids))
    scope = ("config_sha256", "public_model", "recipe", "strong_model")
    for ident, row in comparisons.items():
        left, right = (measurements.get(row[k], {}) for k in ("baseline", "candidate"))
        if (
            ident not in selected
            and all(left.get(k) == base[k] for k in scope)
            and right.get("config_sha256") in candidate_hashes
            and not pending_assignment(left, released)
            and not pending_assignment(right, released)
        ):
            selected.append(ident)
    for ident in selected:
        record = comparisons.get(ident)
        check(
            record and record["split"] == "dev",
            "expected an existing development comparison",
        )
        left, right = (ledger.measure(record[key]) for key in ("baseline", "candidate"))
        check(
            all(left[k] == base[k] for k in scope),
            "comparison belongs to another parent config or entrypoint",
        )
        for row in (left, right):
            independent_dev_row(row, released)
        comparable(left, right)
        paired = {}
        for task, first in left["cases"].items():
            second = right["cases"][task]
            paired[task] = {
                "parent_pass": first["correct"],
                "candidate_pass": second["correct"],
                "same_dev_controls": current_controls.get(task) == control_hash(first),
                "paired_seconds_saved": (
                    first["seconds"] - second["seconds"]
                    if first["correct"] and second["correct"]
                    else None
                ),
            }
        if any(p["parent_pass"] and not p["candidate_pass"] for p in paired.values()):
            action = "protect_regressed_task"
        elif not any(p["parent_pass"] or p["candidate_pass"] for p in paired.values()):
            action = "reproduce_passing_parent"
        elif record.get("disposition") == "advance-to-holdout":
            action = "confirm_frozen_candidate"
        elif record.get("disposition") == "keep-baseline":
            action = "retain_parent"
        else:
            action = "review_paired_development_result"
        matches = {task for task, p in paired.items() if p["same_dev_controls"]}
        regressed = any(
            p["same_dev_controls"] and p["parent_pass"] and not p["candidate_pass"]
            for p in paired.values()
        )
        # A matching regression protects that task. Retirement or promotion of
        # an entire configuration requires the current comparison cohort.
        applies = regressed or (
            bool(matches) and matches == set(paired) == set(current_controls)
        )
        feedback.append(
            {
                "comparison": ident,
                "candidate_sha256": right["config_sha256"],
                "applies_to_current_study": applies,
                "disposition": record.get("disposition"),
                "next_action": action,
                "tasks": paired,
            }
        )
    return feedback


def rank_probes(space, tasks, cards, feedback, depth_observations=None):
    fast, strong = space["fast"], space["strong"]
    choices, evidence, groups = [], {}, defaultdict(list)
    for candidate in space["candidates"]:
        key = candidate["id"]
        predicate = candidate.get("predicate") or {}
        probe = {
            **predicate,
            "id": key,
            "feature": predicate.get("feature", candidate["kind"]),
        }
        if probe["feature"] not in (
            "all_messages",
            "tool_results",
            "fixed_strong",
            "semantic_depth",
        ):
            probe["feature"] = "unknown"
        entries = []
        for info, raw in tasks:
            trace = replay_task(
                raw,
                public_model=space["public_model"],
                strong=strong,
                fast=fast,
                candidate=probe,
                fast_context=cards[fast].get("context_window_size"),
                fast_max_output=cards[fast].get("max_output_tokens"),
                skip_user_messages_matching=space["filters_user_messages"],
                backend_aliases={m: upstream(cards[m]) for m in (strong, fast)},
                task_seconds=info["parent_seconds"],
                depth_observations=(depth_observations or {}).get(
                    (info["measurement"], info["task"]), {}
                ),
            )
            limits = candidate.get("reference_fast_limits")
            if limits:
                old_fit = defaultdict(list)
                for call in trace["calls"]:
                    token_evidence = call["fast_token_fit"]
                    if token_evidence:
                        state = fit_status(
                            token_evidence["requested_output_tokens"],
                            token_evidence["source_prompt_tokens"],
                            limits["context_window_size"],
                            limits["max_output_tokens"],
                        )
                        old_fit[state].append(call["call"])
                trace["fit_under_reference_declarations"] = dict(old_fit)
            entries.append({**info, **trace})
        evidence[key] = entries
        if key == "parent" or probe["feature"] in ("unknown", "fixed_strong"):
            continue
        if all(not e["unknown_calls"] for e in entries):
            signature = tuple(
                (e["measurement"], e["task"], tuple(e["matched_calls"]))
                for e in entries
            )
            groups[signature].append(key)
        retired = any(
            f["applies_to_current_study"]
            and f["candidate_sha256"] == candidate["candidate_sha256"]
            and (
                f["disposition"] in ("keep-baseline", "advance-to-holdout")
                or f["next_action"] == "protect_regressed_task"
            )
            for f in feedback
        )
        if retired:
            continue
        repetitions = defaultdict(list)
        for observed in entries:
            repetitions[observed["task"]].append(observed)
        for task, samples in repetitions.items():
            if any(
                not s["parent_strict"]
                or s["parent_termination"] != "natural_completion"
                or s["opportunity_fraction"] is None
                or (probe["feature"] == "semantic_depth" and s["unknown_calls"])
                for s in samples
            ):
                continue
            exposed = [s for s in samples if s["opportunity_calls"]]
            if not exposed:
                continue
            elapsed = sum(s["opportunity_seconds"] for s in samples)
            duration = sum(s["parent_seconds"] for s in samples)
            fraction = elapsed / duration
            example = min(
                exposed,
                key=lambda s: (
                    abs(s["opportunity_fraction"] - fraction),
                    s["measurement"],
                ),
            )
            choices.append(
                {
                    "candidate": key,
                    "task": task,
                    "measurement": example["measurement"],
                    "anchor": candidate.get("anchor", False),
                    "candidate_sha256": candidate["candidate_sha256"],
                    "affected_parent_calls": example["opportunity_calls"],
                    "measurements": [s["measurement"] for s in samples],
                    "observed_seconds": elapsed,
                    "fraction_of_task": fraction,
                    "parent_seconds": duration,
                }
            )
    # Prefer a candidate the caller can actually send to a public probe. Order by the
    # fraction of parent time exposed per complete episode, then absolute exposure.
    choices.sort(
        key=lambda r: (
            not r["anchor"],
            not bool(r["candidate_sha256"]),
            -r["fraction_of_task"],
            -r["observed_seconds"],
            r["parent_seconds"],
            r["candidate"],
            r["task"],
        )
    )
    ranked, seen = [], set()
    for choice in choices:
        identity = (
            choice["candidate_sha256"] or choice["candidate"],
            choice["task"],
            choice["measurement"],
        )
        if identity not in seen:
            ranked.append(choice)
            seen.add(identity)
    return (
        evidence,
        ranked,
        [members for members in groups.values() if len(members) > 1],
    )


def next_probe(choices, evidence, space, feedback):
    anchor = next((c for c in space["candidates"] if c.get("anchor")), None)
    relevant = [f for f in feedback if f["applies_to_current_study"]]
    anchor_feedback = [
        f
        for f in relevant
        if anchor and f["candidate_sha256"] == anchor["candidate_sha256"]
    ]
    rows = [item for items in evidence.values() for item in items]
    passing = [
        r
        for r in rows
        if r["parent_strict"] and r["parent_termination"] == "natural_completion"
    ]
    if not passing:
        return {
            "action": "reproduce_passing_parent",
            "reason": "no naturally completed strict parent success",
            "tasks": sorted({r["task"] for r in rows}),
        }
    if anchor and not anchor_feedback and not any(c["anchor"] for c in choices):
        return {
            "action": "diagnose_reference",
            "candidate": anchor["id"],
            "reason": "the nominated reference lacks a rankable exposure on a passing parent",
            "checks": [
                "check reference model mapping and declared capacity",
                "find a passing development prefix where the reference can change delivery",
            ],
            "reference_state": anchor["reason"],
        }
    if not choices:
        candidates = {
            c["candidate_sha256"]: c
            for c in space["candidates"]
            if c["candidate_sha256"]
        }
        for result in reversed(anchor_feedback or relevant):
            reviewed = candidates.get(result["candidate_sha256"])
            if reviewed and result["next_action"] != "review_paired_development_result":
                same_candidate = [
                    f
                    for f in relevant
                    if f["candidate_sha256"] == result["candidate_sha256"]
                ]
                outcomes = {f["next_action"] for f in same_candidate}
                if "confirm_frozen_candidate" in outcomes and outcomes.intersection(
                    {
                        "protect_regressed_task",
                        "retain_parent",
                        "reproduce_passing_parent",
                    }
                ):
                    return {
                        "action": "repeat_conflicting_comparisons",
                        "candidate": reviewed["id"],
                        "comparisons": [f["comparison"] for f in same_candidate],
                        "reason": "matching development comparisons disagree; reproduce the pair before promotion",
                    }
                return {
                    "action": result["next_action"],
                    "candidate": reviewed["id"],
                    "comparison": result["comparison"],
                    "tasks": result["tasks"],
                    "reason": "use the verified same-candidate paired outcome before repeating the experiment",
                }
        if any(e["hard_blocked_calls"] for e in rows):
            return {
                "action": "inspect_backend_limits",
                "reason": "matched requests exceed a declared limit",
            }
        opaque = next(
            (
                c
                for c in space["candidates"]
                if c["kind"] == "requires_probe"
                and c["candidate_sha256"]
                and c["id"] != "parent"
            ),
            None,
        )
        if opaque:
            parent = min(passing, key=lambda r: (r["parent_seconds"], r["task"]))
            return {
                "action": "compare_gateway_decisions",
                "candidate": opaque["id"],
                "candidate_sha256": opaque["candidate_sha256"],
                "task": parent["task"],
                "measurement": parent["measurement"],
                "reason": opaque["reason"],
                "probe_checks": [
                    "send the same boundary requests through parent and candidate",
                    "compare winning decision, qualified backend and effective parameters",
                ],
            }
        sites = space["inventory"].get("semantic_thresholds")
        if sites:
            return {
                "action": "collect_depth_observations",
                "signals": [site["signal"] for site in sites],
                "reason": space.get("depth_evidence", {}).get(
                    "reason",
                    "need complete development scores, competing decisions and fast eligibility for a passing task",
                ),
                "input": "--depth-evidence",
            }
        if any(c["kind"] == "requires_probe" for c in space["candidates"]):
            return {
                "action": "inspect_reference_on_gateway",
                "reason": "reference rules need a mapped live preview",
            }
        return {
            "action": "collect_discriminating_prefix",
            "reason": "no ranked opportunity on a passing parent",
            "previous_comparisons": [f["comparison"] for f in feedback],
        }
    best = choices[0]
    sample = next(
        e
        for e in evidence[best["candidate"]]
        if (e["task"], e["measurement"]) == (best["task"], best["measurement"])
    )
    other_passes = sorted(
        {
            e["task"]
            for e in evidence[best["candidate"]]
            if e["task"] != best["task"]
            and e["parent_strict"]
            and e["instruction_sha256"] != sample["instruction_sha256"]
            and e["parent_termination"] == "natural_completion"
            and (
                "changed_decision_calls" not in e
                or (not e["unknown_calls"] and not e["changed_decision_calls"])
            )
        }
    )
    return {
        **best,
        "action": (
            "probe_public_route" if best["candidate_sha256"] else "adapt_reference"
        ),
        "reason": (
            "probe the nominated reference before tuning alternatives"
            if best["anchor"]
            else "ranked by parent request-time share on strict passes among materializable edits first"
        ),
        "guard_options": other_passes,
        "guard_status": (
            "verify_repeatability"
            if other_passes
            else "find_distinct_reproducible_guard"
        ),
        "context_estimate_exceeded": sample["fit_estimate_exceeded_calls"],
        "missing_fit_evidence": sample["missing_fit_calls"],
        "probe_checks": [
            "validate config with target CLI and Router",
            "test below and inside the boundary",
            "verify actual fast response, tool handling, priority, fit and fallback",
        ],
    }


def routing_projection(raw):
    """Allow isolated upstream endpoints and credentials to move without changing route controls."""
    doc = yaml.safe_load(raw)
    for card in doc["providers"]["models"]:
        for backend in card.get("backend_refs", []):
            backend.pop("endpoint", None)
            backend.pop("auth", None)
    return doc


def after_gateway(path, step, space, evidence, cards, source_config, projected_config):
    if __package__:
        from .planning_probe import assess_probe
    else:
        from planning_probe import assess_probe
    check(
        step["action"] == "probe_public_route",
        "no current candidate probe to reconcile",
    )
    c = next(c for c in space["candidates"] if c["id"] == step["candidate"])
    fast, strong = space["fast"], space["strong"]
    check(
        c.get("public_probe_predicate"),
        "this action needs a gateway comparison specific to its decision rules",
    )
    probe_predicate = {
        **c["public_probe_predicate"],
        "skip_user_messages_matching": space["filters_user_messages"],
    }
    fast_limit, strong_limit = (
        cards[m].get("context_window_size") for m in (fast, strong)
    )
    require_fallback = (
        type(fast_limit) is int
        and type(strong_limit) is int
        and strong_limit > fast_limit > 0
    )
    if probe_predicate["feature"] == "semantic_depth":
        require_fallback = (
            require_fallback
            and strong in probe_predicate["bindings"][probe_predicate["fallback"]]
        )
    # ModelCapabilities accepts "vision" as an alias for OpenAI image input.
    image_supported = any(
        str(cap).strip().lower() in {"vision", "image_input"}
        for model in (fast, strong)
        for cap in cards[model].get("capabilities", [])
    )
    output_limits = [cards[model].get("max_output_tokens") for model in (fast, strong)]
    report = assess_probe(
        path,
        candidate_sha=step["candidate_sha256"],
        predicate=probe_predicate,
        public_model=space["public_model"],
        fast=fast,
        strong=strong,
        backend_aliases={m: upstream(cards[m]) for m in (strong, fast)},
        require_context_fallback=require_fallback,
        require_image_refusal=not image_supported,
        require_output_refusal=all(
            type(limit) is int and limit > 0 for limit in output_limits
        ),
    )
    projected_sha = report.get("projected_sha")
    issue = None
    if projected_config is not None:
        deployed = projected_config.read_bytes()
        if digest(deployed) != projected_sha or routing_projection(
            deployed
        ) != routing_projection(source_config):
            issue = (
                "failed",
                "deployed configuration differs from the receipt or the target routing policy",
            )
    elif projected_sha != digest(source_config):
        issue = (
            "incomplete",
            "supply --probe-config to check the isolated deployment against the candidate",
        )
    if issue:
        severity, reason = issue
        report["issues"].append(
            {
                "case": "receipt",
                "field": "projected_sha",
                "severity": severity,
                "reason": reason,
            }
        )
        if report["status"] != "failed":
            report["status"] = severity
    report["deployment_config_verified"] = issue is None
    if report["status"] == "real_pass":
        observations = defaultdict(list)
        for e in evidence[c["id"]]:
            if e["task"] in step["guard_options"]:
                observations[e["task"]].append(e)
        repeatable = [
            t
            for t, rows in observations.items()
            if len(rows) >= 2
            and all(
                r["parent_strict"]
                and r["parent_termination"] == "natural_completion"
                and (
                    probe_predicate["feature"] != "semantic_depth"
                    or (not r["unknown_calls"] and not r["changed_decision_calls"])
                )
                for r in rows
            )
        ]
        if repeatable:
            action, reason = (
                "plan_paired_development",
                "real boundary verified and a separate parent guard reproduced",
            )
        elif step["guard_options"]:
            action, reason = (
                "verify_guard_repeatability",
                "confirm the separate parent control before spending on the candidate",
            )
        else:
            action, reason = (
                "find_distinct_reproducible_guard",
                "no separate passing guard in the supplied development records",
            )
        step["reproduced_guard_options"] = sorted(repeatable)
    elif report["status"] == "fixture_pass":
        action, reason = (
            "probe_real_backends",
            "fixture responses verify the gateway branch; test actual backend admission and tools",
        )
    elif report["status"] == "failed":
        action, reason = (
            "diagnose_gateway_probe",
            "inspect the failed public probe before funding a whole episode",
        )
    else:
        action, reason = (
            "complete_gateway_probe",
            "the receipt does not cover the required public checks",
        )
    return {**step, "action": action, "reason": reason, "public_probe": report}


class DevelopmentSnapshot:
    """Keep enrollment and later readback on the same development ledger prefix."""

    def __init__(self, ledger, through=None):
        self.path = ledger.path
        rows = [r for r in ledger.rows() if r["split"] == "dev"]
        if through is not None:
            ids = [r["record_id"] for r in rows]
            check(through in ids, "development plan ledger checkpoint is unavailable")
            rows = rows[: ids.index(through) + 1]
        self._rows = rows
        self.through = rows[-1]["record_id"] if rows else None

    def rows(self):
        return self._rows

    def measure(self, ident):
        for row in self._rows:
            if row["record_id"] == ident and row["kind"] == "measurement":
                return row
        raise ValueError(f"unknown development measurement: {ident}")


def replay_inputs(args):
    """Store read-only inputs. Publishing a config is never replayed."""
    filenames = ("anchor", "reference_library", "probe_receipt", "probe_config")
    lists = ("reference", "edit_space", "schedule_card")

    def absolute(f):
        return str(Path(f).resolve())

    return {
        "base": args.base,
        "fast": args.fast,
        **(
            {"depth_evidence": absolute(args.depth_evidence)}
            if getattr(args, "depth_evidence", None)
            else {}
        ),
        **(
            {"anchor_models": args.anchor_models}
            if getattr(args, "anchor_models", None)
            else {}
        ),
        "prefixes": args.prefixes,
        "comparison": args.comparison,
        **{
            key: absolute(getattr(args, key)) if getattr(args, key) else None
            for key in filenames
        },
        **{key: [absolute(p) for p in getattr(args, key)] for key in lists},
        "reference_entry": [[entry, absolute(p)] for entry, p in args.reference_entry],
        "anchor_entry": (
            [args.anchor_entry[0], absolute(args.anchor_entry[1])]
            if args.anchor_entry
            else None
        ),
    }


def replay_plan(ledger, report):
    checkpoint = report["replay"]
    snapshot = DevelopmentSnapshot(ledger, checkpoint["through"])
    args = SimpleNamespace(**checkpoint["inputs"])
    if getattr(args, "depth_evidence", None):
        args.depth_evidence = Path(args.depth_evidence)
    for key in ("anchor", "reference_library", "probe_receipt", "probe_config"):
        value = getattr(args, key)
        setattr(args, key, Path(value) if value is not None else None)
    for key in ("reference", "edit_space", "schedule_card"):
        setattr(args, key, [Path(p) for p in getattr(args, key)])
    args.cmd, args.out_candidate = "plan-dev", None
    return run(snapshot, args)


def run(ledger, args):
    if __package__:
        from .loop import baseline_config, publish_config, released_assignments
    else:
        from loop import baseline_config, publish_config, released_assignments
    if args.cmd == "plan-space":
        raw = args.config.read_bytes()
        space, _ = inventory_for(
            yaml.safe_load(raw), digest(raw), args.public_model, args.strong, args
        )
        return space
    if not isinstance(ledger, DevelopmentSnapshot):
        ledger = DevelopmentSnapshot(ledger)
    base = ledger.measure(args.base)
    check(
        not args.probe_config or args.probe_receipt,
        "--probe-config requires --probe-receipt",
    )
    released = released_assignments(ledger, args.schedule_card)
    tasks = observed_tasks(ledger, base, args.prefixes, released)
    doc = baseline_config(base)
    space, configs = inventory_for(
        doc, base["config_sha256"], base["public_model"], base["strong_model"], args
    )
    cards = model_cards(doc)
    depth_observations = None
    if getattr(args, "depth_evidence", None):
        depth_observations = add_candidate(
            args.depth_evidence, doc, space, configs, tasks, cards
        )
    hashes = {
        c["candidate_sha256"] for c in space["candidates"] if c["candidate_sha256"]
    }
    current_controls = {info["task"]: info["control_sha256"] for info, _ in tasks}
    feedback = prior_results(
        ledger, args.comparison, base, hashes, released, current_controls
    )
    evidence, choices, groups = rank_probes(
        space, tasks, cards, feedback, depth_observations
    )
    step = next_probe(choices, evidence, space, feedback)
    proposal = step
    if args.probe_receipt:
        check(
            step.get("candidate") in configs,
            "no materialized candidate for this probe receipt",
        )
        step = after_gateway(
            args.probe_receipt,
            step,
            space,
            evidence,
            cards,
            configs[step["candidate"]],
            args.probe_config,
        )
    if args.out_candidate:
        check(
            proposal["action"] in ("probe_public_route", "compare_gateway_decisions")
            and proposal.get("candidate") in configs,
            "no recommended materializable candidate to publish",
        )
        raw = configs[proposal["candidate"]]
        check(
            publish_config(args.out_candidate, raw.decode("utf-8"))
            == proposal["candidate_sha256"],
            "candidate publication identity mismatch",
        )
    return {
        "schema": "recipe-lab-development-plan-v1",
        "based_on": args.base,
        "space": space,
        "replay": {"through": ledger.through, "inputs": replay_inputs(args)},
        "prefix_evidence": evidence,
        "prior_comparisons": feedback,
        "ranked_probes": choices,
        "same_observed_predicate": groups,
        "next": step,
        "decision_rules": {
            "predicate_misses": "Check visible input or choose a development task with coverage.",
            "matched_but_no_fast_response": "Inspect priority overrides, model eligibility and fallback.",
            "passing_parent_regresses": "Retain parent; narrow the handoff or introduce a tested guard.",
            "parent_does_not_reproduce": "Repeat or replace the development control before attributing a loss.",
            "joint_success": "Compare paired DSH wall time; confirm on independent tasks before promotion.",
        },
        "timing_scope": "Observed source request intervals; changing a response may change every later turn.",
        "dispatch": False,
    }
