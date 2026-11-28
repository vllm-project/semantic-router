"""Replay a keep-parent decision on verified, time-bounded development evidence."""

from math import fsum
from pathlib import Path

if __package__:
    from .loop_evidence import check, digest
    from .schedule import (
        canonical,
        decision,
        episode_ids,
        parse,
        protocol,
        replay_attempts,
    )
else:
    from loop_evidence import check, digest
    from schedule import (
        canonical,
        decision,
        episode_ids,
        parse,
        protocol,
        replay_attempts,
    )


def add_arguments(parser):
    parser.add_argument(
        "--point",
        nargs=2,
        action="append",
        required=True,
        metavar=("CARD", "VISIBLE_ATTEMPTS"),
        help="one development checkpoint per enrolled study",
    )
    parser.add_argument("--before-skill", type=Path, required=True)
    parser.add_argument("--after-skill", type=Path, required=True)
    parser.add_argument(
        "--revision-source",
        action="append",
        default=[],
        help="development comparison that motivated the proposed edit",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        help="paired agent choices bound to the exported replay packet",
    )


def fingerprint(value):
    return digest(canonical(value).encode("utf-8"))


def revision_evidence(ledger, ids, known_episodes):
    """Validate source chronology before opening its task evidence."""
    if __package__:
        from .loop import comparable, verify
    else:
        from loop import comparable, verify

    if not ids:
        return
    comparisons = {
        r["record_id"]: r for r in ledger.rows() if r["kind"] == "comparison"
    }
    check(len(set(ids)) == len(ids), "duplicate revision source")
    for ident in ids:
        source = comparisons.get(ident)
        check(
            source and source["split"] == "dev",
            "revision source must be a dev comparison",
        )
        pair = [ledger.measure(source[role]) for role in ("baseline", "candidate")]
        check(
            all(r["split"] == "dev" for r in pair),
            "revision source links non-dev evidence",
        )
        for row in pair:
            episodes = episode_ids(row)
            check(
                episodes and episodes <= known_episodes,
                "revision source contains a future episode",
            )
        for row in pair:
            verify(row)
        comparable(*pair)


def read_point(ledger, card_path, count):
    card, card_hash, report, parent = protocol(ledger, card_path)
    state_path = ledger.path.parent / "schedules" / card["study"] / "state.json"
    state = parse(state_path.read_bytes())
    check(state["card_sha256"] == card_hash, "experiment card changed after enrollment")
    check(
        0 <= count <= len(state["attempts"]),
        "cutoff outside the recorded development history",
    )
    prefix = dict(state, attempts=state["attempts"][:count])
    past = replay_attempts(ledger, card, report, parent, prefix)
    check(
        decision(card, report, past)["status"] == "run",
        "checkpoint already has a terminal decision",
    )
    visible = []
    for attempt in past:
        item = {
            k: attempt[k]
            for k in ("repeat", "role", "task", "valid", "seconds", "episode_sha256")
        }
        if item["valid"]:
            item["strict"] = attempt["strict"]
        else:
            item["invalid_reason"] = attempt["reason"]
        visible.append(item)
    point = {
        "study": card["study"],
        "enrollment": state["enrollment"],
        "card_sha256": card_hash,
        "visible_attempts": count,
        "split": "dev",
        "history": visible,
        "guard_task": card["guard_task"],
        "target_task": card["target_task"],
        "budget_episodes": card["budget_episodes"],
        "repeats": card["repeats"],
        "action": report["action"],
        "candidate_sha256": report["candidate_sha256"],
        "prefix_measurement": report["prefixes"],
        "parent_measurement": parent["record_id"],
        "fixed_prefixes": {
            t: {k: s[k] for k in ("changed", "unknown", "observed")}
            for t, s in report["by_task"].items()
        },
    }
    # The returned agent packet contains prospectively bound fields only. A future
    # attempt can change the score but cannot change any byte in this packet.
    return point, (card, report, parent, state)


def validate_choices(file, packet):
    choices = parse(file.read_bytes())
    check(
        isinstance(choices, dict)
        and set(choices) == {"packet_sha256", "choices"}
        and choices["packet_sha256"] == packet["packet_sha256"],
        "decisions must match the frozen skills and replay packet",
    )
    rows = choices["choices"]
    check(
        isinstance(rows, dict) and rows.keys() == packet["points"].keys(),
        "supply both decisions for every frozen checkpoint",
    )
    for pair in rows.values():
        check(
            isinstance(pair, dict) and set(pair) == {"before", "after"},
            "each checkpoint requires old and revised skill decisions",
        )
        for side, choice in pair.items():
            check(
                isinstance(choice, dict)
                and set(choice) == {"choice", "skill_sha256"}
                and choice["skill_sha256"] == packet["skills"][side]
                and choice["choice"] in ("continue", "keep_parent"),
                "invalid choice or wrong skill identity",
            )
    return rows


def freeze_choices(workspace, packet, choices):
    if __package__:
        from .loop import publish_config
    else:
        from loop import publish_config

    folder = workspace / "skill-replays"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (packet["packet_sha256"] + ".json")
    text = canonical({"packet": packet, "decisions": choices}) + "\n"
    try:
        publish_config(path, text)
    except FileExistsError:
        check(
            path.read_bytes() == text.encode(),
            "paired decisions already frozen for this packet",
        )
    return digest(text.encode())


def score_point(ledger, context, point, choices):
    card, report, parent, state = context
    history = replay_attempts(ledger, card, report, parent, state)
    outcome = decision(card, report, history)["status"]
    row = {
        "study": point["study"],
        "observed_outcome": outcome,
        "checkpoint_sha256": fingerprint(point),
        "visible_attempts": point["visible_attempts"],
    }
    if outcome not in ("advance", "revise", "drop"):
        return dict(row, scored=False)
    tail = history[point["visible_attempts"] :]
    for side, choice in choices.items():
        proceed = choice["choice"] == "continue"
        row[side] = {
            "choice": choice["choice"],
            "same_selection": proceed or outcome != "advance",
            "remaining_episodes": len(tail) if proceed else 0,
            "remaining_recorded_seconds": (
                fsum(a["seconds"] for a in tail) if proceed else 0.0
            ),
        }
    return dict(row, scored=True)


def run(ledger, args):
    skills = {}
    for side, path in (("before", args.before_skill), ("after", args.after_skill)):
        raw = path.read_bytes()
        check(raw.decode("utf-8-sig").strip(), "skill must contain UTF-8 text")
        skills[side] = digest(raw)
    check(skills["before"] != skills["after"], "supply two distinct skill versions")
    points, contexts = {}, {}
    for name, count in args.point:
        point, context = read_point(ledger, Path(name), int(count))
        study = point["study"]
        check(study not in points, "each study may contribute one checkpoint")
        known = set(context[3]["historical_episodes"])
        known.update(h["episode_sha256"] for h in point["history"])
        revision_evidence(ledger, args.revision_source, known)
        points[study], contexts[study] = point, context
    packet = {
        "schema": "recipe-lab-skill-replay-v1",
        "skills": skills,
        "points": points,
        "revision_sources": sorted(args.revision_source),
        "choices": ["continue", "keep_parent"],
        "objective": "Preserve whether the completed development study advances the candidate, then reduce remaining episodes.",
    }
    packet["packet_sha256"] = fingerprint(packet)
    if not args.decisions:
        return packet
    choices = validate_choices(args.decisions, packet)
    submission = freeze_choices(ledger.path.parent, packet, choices)
    rows = [score_point(ledger, contexts[s], p, choices[s]) for s, p in points.items()]
    scored = [r for r in rows if r["scored"]]
    regressions = sum(
        r["before"]["same_selection"] and not r["after"]["same_selection"]
        for r in scored
    )
    corrected = sum(
        r["after"]["same_selection"] and not r["before"]["same_selection"]
        for r in scored
    )
    saved = sum(
        r["before"]["remaining_episodes"] - r["after"]["remaining_episodes"]
        for r in scored
    )
    seconds = fsum(
        r["before"]["remaining_recorded_seconds"]
        - r["after"]["remaining_recorded_seconds"]
        for r in scored
    )
    if regressions:
        verdict = "reject_quality_regression"
    elif len(scored) < len(rows):
        verdict = "incomplete_development_outcomes"
    elif any(not r["after"]["same_selection"] for r in scored):
        verdict = "revised_skill_loses_candidate"
    elif corrected or saved > 0:
        verdict = (
            "positive_historical_utility"
            if args.revision_source
            else "positive_needs_revision_source"
        )
    else:
        verdict = "retain_current_skill"
    return {
        "schema": "recipe-lab-skill-replay-score-v1",
        "packet_sha256": packet["packet_sha256"],
        "frozen_decisions_sha256": submission,
        "skills": skills,
        "revision_sources": packet["revision_sources"],
        "by_study": rows,
        "scored_studies": len(scored),
        "new_selection_errors": regressions,
        "corrected_selections": corrected,
        "counterfactual_episodes_saved": saved,
        "counterfactual_recorded_seconds_saved": seconds,
        "verdict": verdict,
        "scope": "Historical utility of early keep-parent decisions. Evaluate recipe quality and cross-project transfer separately.",
    }
