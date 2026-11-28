"""Prospectively enroll the trial slots of a bounded search and bind proposer usage."""

import secrets
from contextlib import ExitStack
from pathlib import Path

if __package__:
    from . import search_protocol
    from .loop_evidence import check, component, digest
    from .schedule import canonical, parse
else:
    import search_protocol
    from loop_evidence import check, component, digest
    from schedule import canonical, parse


def io():
    if __package__:
        from . import loop
    else:
        import loop
    return loop


def add_arguments(subs):
    parser = subs.add_parser(
        "search-enroll", help="reserve ordered trial slots before starting the search"
    )
    search_protocol.add_arguments(parser)
    parser.add_argument("--allocation", type=Path, required=True)


def enrollment_root(ledger):
    return ledger.path.parent / "search-enrollments"


def enroll(ledger, args):
    report = search_protocol.preflight(ledger, args)
    meta = report["comparison"]
    check(
        report["status"] == "target_actions_distinguished"
        and meta["target_selection_headroom"] == "non_exhaustive",
        "a ready, non-exhaustive development preflight is required",
    )
    spec = parse(args.allocation.read_bytes())
    check(
        isinstance(spec, dict)
        and set(spec) == {"schema", "study", "protocol_sha256", "methods"}
        and spec["schema"] == "recipe-lab-search-allocation-v1"
        and spec["study"] == report["study"]
        and spec["protocol_sha256"] == report["protocol_sha256"],
        "allocation differs from search protocol",
    )
    methods = spec["methods"]
    check(
        isinstance(methods, dict) and set(methods) == set(search_protocol.METHODS),
        "allocate all three search methods",
    )
    slots = []
    for ids in methods.values():
        check(
            isinstance(ids, list)
            and len(ids) == meta["candidate_trials_per_method"]
            and all(component(i) for i in ids),
            "allocate an ordered name for every trial slot",
        )
        slots.extend(ids)
    check(len(set(slots)) == len(slots), "trial slot names must be unique")
    configs = {
        a["upper"]: a["config_sha256"] for a in report["target_development"]["actions"]
    }
    identity = {
        "schema": "recipe-lab-search-enrollment-v1",
        "study": component(spec["study"]),
        "protocol_sha256": spec["protocol_sha256"],
        "preflight_sha256": digest(canonical(report).encode()),
        "split_plans": meta["split_plans"],
        "space_sha256": report["space_sha256"],
        "budget": meta["shared_budget_per_method"],
        "methods": methods,
        "eligible_candidates": [
            configs[u] for u in meta["shared_class_representatives"]
        ],
        "random_candidates": [configs[u] for u in meta["random_class_order"]],
    }
    root = enrollment_root(ledger)
    root.mkdir(parents=True, exist_ok=True)
    api = io()
    # A native schedule takes its own writer lock before checking the registry.
    # Acquire in the same order so an ordinary schedule that saw no enrollment
    # either finishes creating its state first, or observes the reserved slot.
    with ExitStack() as locks:
        for trial in sorted(slots):
            schedule_folder = ledger.path.parent / "schedules" / trial
            schedule_folder.mkdir(parents=True, exist_ok=True)
            locks.enter_context(api.writer_lock(schedule_folder / "writer.lock"))
        locks.enter_context(api.writer_lock(root / "registry.lock"))
        folder = root / identity["study"]
        path = folder / "enrollment.json"
        if path.exists():
            existing = parse(path.read_bytes())
            check(
                {k: v for k, v in existing.items() if k != "enrollment"} == identity,
                "search already enrolled",
            )
            return existing
        for other in root.glob("*/enrollment.json"):
            allocated = parse(other.read_bytes())["methods"]
            check(
                not any(set(slots).intersection(v) for v in allocated.values()),
                "trial slot already belongs to another study",
            )
        check(
            not any(
                (ledger.path.parent / "schedules" / i / "state.json").exists()
                for i in slots
            ),
            "trial already started before search enrollment",
        )
        folder.mkdir(exist_ok=True)
        result = dict(identity, enrollment=secrets.token_hex(32))
        api.publish_config(path, canonical(result) + "\n")
        return result


def load_enrollment(ledger, report):
    path = enrollment_root(ledger) / component(report["study"]) / "enrollment.json"
    check(path.is_file(), "search must be enrolled before its trials")
    enrollment = parse(path.read_bytes())
    check(
        enrollment["preflight_sha256"] == digest(canonical(report).encode()),
        "enrolled search preflight changed",
    )
    return enrollment


def lookup_trial(root, study):
    for path in root.glob("*/enrollment.json"):
        enrollment = parse(path.read_bytes())
        for method, slots in enrollment["methods"].items():
            if study in slots:
                return path.parent, enrollment, method, slots
    return None


def scoped_budget(card, registration):
    if registration is None:
        return card
    return dict(
        card,
        budget_episodes=min(
            card["budget_episodes"], registration["remaining_method_episodes"]
        ),
    )


def historical_card(ledger, card, card_hash):
    """Reconstruct the same trial budget for replay, summaries and subsequent polls."""
    match = lookup_trial(enrollment_root(ledger), card["study"])
    if match:
        folder, enrollment, _, _ = match
        path = folder / (card["study"] + ".started.json")
        if path.is_file():
            registration = parse(path.read_bytes())
            check(
                registration["card_sha256"] == card_hash
                and registration["enrollment"] == enrollment["enrollment"],
                "registered trial card changed",
            )
            return scoped_budget(card, registration)
    return card


def before_trial(ledger, study, card_hash, candidate_hash):
    root = enrollment_root(ledger)
    if not root.is_dir():
        return None
    api = io()
    with api.writer_lock(root / "registry.lock"):
        match = lookup_trial(root, study)
        if not match:
            return None
        folder, enrollment, method, slots = match
        registered = folder / (study + ".started.json")
        identity = {
            "card_sha256": card_hash,
            "enrollment": enrollment["enrollment"],
            "candidate_sha256": candidate_hash,
        }
        if registered.exists():
            receipt = parse(registered.read_bytes())
            check(
                all(receipt.get(key) == value for key, value in identity.items()),
                "registered trial card changed",
            )
            return receipt
        check(
            not (
                ledger.path.parent
                / "search-studies"
                / enrollment["study"]
                / "freeze.json"
            ).exists(),
            "search selections are already frozen",
        )
        position = slots.index(study)
        remaining = enrollment["budget"]["episodes"]
        for earlier in slots[:position]:
            closed = folder / (earlier + ".closed.json")
            check(
                closed.is_file(),
                "previous trial slot is still open",
            )
            past = parse(closed.read_bytes())
            check(
                past["candidate_sha256"] != candidate_hash,
                "candidate already used by this method",
            )
            remaining -= past["spent_episodes"]
        check(remaining >= 0, "previous trials exceeded the enrolled episode budget")
        check(
            candidate_hash in enrollment["eligible_candidates"],
            "candidate outside enrolled search space",
        )
        if method == "random":
            check(
                candidate_hash == enrollment["random_candidates"][position],
                "trial differs from frozen random order",
            )
        receipt = dict(identity, remaining_method_episodes=remaining)
        api.publish_config(registered, canonical(receipt) + "\n")
        return receipt


def after_trial(ledger, study, card_hash, status, spent):
    root = enrollment_root(ledger)
    if status == "run" or not root.is_dir():
        return
    check(status in ("advance", "drop", "revise", "defer"), "unknown trial outcome")
    api = io()
    with api.writer_lock(root / "registry.lock"):
        match = lookup_trial(root, study)
        if match:
            folder, enrollment, _, _ = match
            start = parse((folder / (study + ".started.json")).read_bytes())
            check(
                start["card_sha256"] == card_hash
                and start["enrollment"] == enrollment["enrollment"]
                and 0 <= spent <= start["remaining_method_episodes"],
                "trial started outside search enrollment",
            )
            closed = folder / (study + ".closed.json")
            receipt = dict(start, status=status, spent_episodes=spent)
            if closed.exists():
                check(parse(closed.read_bytes()) == receipt, "closed trial changed")
            else:
                api.publish_config(closed, canonical(receipt) + "\n")


def check_trial_slots(ledger, enrollment, method, listed_study_ids):
    """Also check native schedule nonces. The search journal alone cannot bind older episodes."""
    slots = enrollment["methods"][method]
    check(
        listed_study_ids == slots[: len(listed_study_ids)],
        "trial list differs from allocated method order",
    )
    folder = enrollment_root(ledger) / enrollment["study"]
    states = []
    for i, study in enumerate(slots):
        state_path = ledger.path.parent / "schedules" / study / "state.json"
        started = folder / (study + ".started.json")
        closed = folder / (study + ".closed.json")
        if i >= len(listed_study_ids):
            check(
                not (started.exists() or state_path.exists() or closed.exists()),
                "started search trial omitted",
            )
            states.append({"study": study, "status": "unused_slot"})
            continue
        check(closed.is_file() and state_path.is_file(), "listed trial is not closed")
        receipt, state = parse(closed.read_bytes()), parse(state_path.read_bytes())
        check(
            receipt["enrollment"]
            == state.get("search_enrollment")
            == enrollment["enrollment"]
            and receipt["card_sha256"] == state["card_sha256"],
            "trial state is not bound to search enrollment",
        )
        check(
            receipt["spent_episodes"] == len(state["attempts"]),
            "closed trial episode count changed",
        )
        states.append(
            {
                "study": study,
                "status": receipt["status"],
                "episodes": len(state["attempts"]),
                "card_sha256": state["card_sha256"],
            }
        )
    return states


def proposal_usage(folder, usage_ref, used, enrollment, method):
    def read(ref):
        check(
            isinstance(ref, dict) and set(ref) == {"file", "sha256"},
            "pin the proposal usage file",
        )
        return parse(search_protocol.pinned_file(folder, ref["file"], ref["sha256"]))

    usage = read(usage_ref)
    check(
        isinstance(usage, dict)
        and set(usage) == {"schema", "enrollment", "method", "calls"}
        and usage["schema"] == "recipe-lab-search-usage-v1"
        and usage["enrollment"] == enrollment["enrollment"]
        and usage["method"] == method
        and isinstance(usage["calls"], list),
        "wrong method usage manifest",
    )
    tokens = 0
    for ref in usage["calls"]:
        call = read(ref)
        check(
            call["schema"] == "recipe-lab-proposal-call-v1"
            and call["enrollment"] == enrollment["enrollment"]
            and call["method"] == method
            and call["trial_study"] in enrollment["methods"][method],
            "proposal call belongs to another enrollment or trial slot",
        )
        response = call["response"]
        key = response["id"]
        check(
            isinstance(key, str) and key.strip() and ("provider-call", key) not in used,
            "missing or reused provider response ID",
        )
        counts = response["usage"]
        check(
            all(
                type(counts[k]) is int and counts[k] >= 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            and counts["prompt_tokens"] + counts["completion_tokens"]
            == counts["total_tokens"],
            "provider token counts missing or inconsistent",
        )
        used.add(("provider-call", key))
        tokens += counts["total_tokens"]
    return tokens
