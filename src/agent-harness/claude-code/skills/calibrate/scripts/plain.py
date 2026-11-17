"""What calibrate tells the user, in plain words: each app, each line, the models, and the
recommendation. No option names here; `--detail` prints those.
"""

from __future__ import annotations

from decisions import TIERS, tier_of
from fitting import MIN_LABELS

# Each app's name, what it does, and the AskUserQuestion header (at most 12 characters).
APP_WORDS = {
    "router": ("Routing", "which model each subagent gets", "Model choice"),
    "guard": ("Risky commands", "asks you before a risky command runs", "Command"),
    "redact": (
        "Personal data",
        "hides personal data from Claude",
        "Private data",
    ),
    "shield": (
        "Injection warnings",
        "warns Claude when a file or page tries to instruct it",
        "Injection",
    ),
    "escalate": (
        "Pushback",
        "sends the next subagent to Opus when you push back",
        "Pushback",
    ),
    "check": (
        "Answer checks",
        "marks answers that claim more than Claude read",
        "Answer check",
    ),
}

# Each threshold's plain name and what it does, in one sentence.
LINES = {
    "routerEasyBelow": ("Easy line", "tasks judged easier than this go to Haiku."),
    "routerHardFrom": (
        "Hard line",
        "tasks judged at least this hard go to Opus; those between the two lines go to Sonnet.",
    ),
    "routerMinConfidence": (
        "Confidence line",
        "vSR changes a subagent's model only when it is at least this sure.",
    ),
    "guardReview": (
        "Review line",
        "vSR asks you before a command it rates at least this risky.",
    ),
    "escalateFeedback": (
        "Complaint line",
        "when your message reads at least this much like a complaint about the last answer, the next subagent goes to Opus and Claude re-checks.",
    ),
    "escalateReask": (
        "Repeat line",
        "when you ask for the same thing again at least this clearly, vSR does the same.",
    ),
    "redactPii": (
        "Hiding line",
        "vSR hides text it is at least this sure is personal data (secrets are always hidden).",
    ),
    "shieldInject": (
        "Warning line",
        "vSR warns Claude about a file or page at least this likely to be trying to instruct it.",
    ),
    "checkUnsupported": (
        "Check line",
        "vSR marks an answer unverified when it is at least this sure the answer claims things Claude did not read.",
    ),
}

# The verdict a binary app logs when it acted.
ACTED_VERDICT = {
    "guard": "held",
    "escalate": "escalated",
    "redact": "hidden",
    "shield": "shielded",
    "check": "unverified",
}

# What the log shows for a binary app: {n} decisions, {acted} acted, {other} not.
LOGGED = {
    "guard": "{n} commands: {acted} held for you, {other} allowed.",
    "escalate": "{n} messages taken as a complaint about the last answer (vSR logs only these).",
    "redact": "{n} texts with personal data hidden (vSR logs only these).",
    "shield": "{n} files or pages Claude was warned about (vSR logs only these).",
    "check": "{n} answers checked: {acted} marked unverified, {other} passed.",
}

# What a change would do on the user's log, for a binary app.
EFFECT = {
    "guard": "vSR would hold {after} of these {n} commands for you{was}.",
    "escalate": "vSR would escalate {after} of these {n} messages{was}.",
    "redact": "vSR would hide data in {after} of these {n} texts{was}.",
    "shield": "vSR would warn Claude about {after} of these {n} files or pages{was}.",
    "check": "vSR would mark {after} of these {n} answers unverified{was}.",
}

# Which labels the log implies for an app, as the user did them.
IMPLIED_BY = {"router": "your pushback implies", "guard": "your declines imply"}

GOAL_WORDS = {
    "balanced": "balanced: as few wrong calls as possible either way",
    "fewer-false-alarms": "fewer false alarms: act only when sure, even if some cases slip by",
    "catch-more": "catch more: act on more borderline cases, even with some false alarms",
}

KEPT = "your session's model"
MORE = "answer more in /vsr or here"


def title(app: str) -> str:
    return f"{APP_WORDS[app][0]} ({APP_WORDS[app][1]})"


def model_name(model: object) -> str:
    """Haiku, Sonnet or Opus as people say them; another model by its name."""
    tier = tier_of(model)
    return TIERS[tier].capitalize() if tier is not None else str(model or KEPT)


def tier_name(tier: str) -> str:
    return KEPT if tier == "kept" else tier.capitalize()


def difficulty(level: float, easy: float, hard: float) -> str:
    """A difficulty level in words, from the tier lines in use."""
    return "easy" if level < easy else ("multi-step" if level < hard else "hard")


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def answered(app: str, explicit: int, implied: int) -> str:
    more = f" (+{implied} {IMPLIED_BY.get(app, 'the log implies')})" if implied else ""
    return f"You answered {explicit}{more}; {MIN_LABELS} are needed to tune."


def router_logged(tiers: dict[str, int]) -> str:
    n = sum(tiers.values())
    parts = [
        f"{tiers[t]} {'went to ' if i == 0 else 'to '}{t.capitalize()}"
        for i, t in enumerate(TIERS)
    ]
    if tiers.get("other"):
        parts.append(f"{tiers['other']} to other models")
    parts.append(f"{tiers['kept']} kept {KEPT}")
    if tiers.get("off"):
        parts.append(f"{tiers['off']} while routing was off")
    return f"{plural(n, 'subagent')}: {', '.join(parts)}."


def router_moves(moves: list[dict]) -> str:
    """What new router lines would do to the user's logged subagents, in words."""
    phrases = []
    order = (*TIERS, "kept")
    moves = sorted(moves, key=lambda m: (order.index(m["from"]), order.index(m["to"])))
    routed = [m for m in moves if m["from"] == "kept"]
    for m in moves:
        if m["from"] == "kept":
            continue
        went = f"{plural(m['n'], 'task')} that went to {tier_name(m['from'])}"
        if m["to"] == "kept":
            phrases.append(f"{went} would keep {KEPT}")
        else:
            phrases.append(f"{went} would go to {tier_name(m['to'])}")
    if routed:
        n = sum(m["n"] for m in routed)
        where = ", ".join(f"{m['n']} to {tier_name(m['to'])}" for m in routed)
        phrases.append(
            f"{plural(n, 'subagent')} kept on {KEPT} would be routed ({where})"
        )
    if not phrases:
        return "no subagent in your log would get a different model."
    return "; ".join(phrases) + "."


RECOMMEND = {
    "apply": "Recommended: apply",
    "keep": "Recommended: keep the current settings",
}


def recommendation(entry: dict) -> str:
    if entry["recommend"] == "more":
        return f"Not enough answers yet: {entry['labels']} of {MIN_LABELS} ({MORE})"
    return RECOMMEND[entry["recommend"]]


def app_suggestion(app: str, entry: dict) -> list[str]:
    out = [title(app)]
    if "before" not in entry:
        return [*out, f"  {entry['plain']}"]
    before, after = entry["before"]["values"], entry["after"]["values"]
    for key, was in before.items():
        name, does = LINES[key]
        now = after[key]
        moved = (
            f"{was:.2f} → {now:.2f}" if abs(now - was) > 1e-9 else f"stays {was:.2f}"
        )
        out.append(f"  {name} {moved}: {does}")
    if entry["recommend"] != "more":
        if app == "router":
            effect = router_moves(entry["moves"])
            effect = effect[0].upper() + effect[1:]
        else:
            was, now = entry["before"]["acted"], entry["after"]["acted"]
            effect = EFFECT[app].format(
                n=entry["decisions"],
                after=now,
                was=f" instead of {was}" if was != now else " (the same as now)",
            )
        out.append(f"  On your log: {effect}")
        out.append(
            f"  Your answers: {entry['agreeBefore']} of {entry['labels']} agree with vSR now, "
            f"{entry['agreeAfter']} of {entry['labels']} would after."
        )
    out.append(f"  {recommendation(entry)}")
    return out


def suggestion(plan: dict) -> str:
    out = []
    binary = [app for app in plan["apps"] if app != "router"]
    if binary and plan.get("goal"):
        out += [f"Aim: {GOAL_WORDS[plan['goal']]}.", ""]
    for app, entry in plan["apps"].items():
        out += [*app_suggestion(app, entry), ""]
    apply = [app for app, e in plan["apps"].items() if e.get("recommend") == "apply"]
    if len(plan["apps"]) > 1:
        if apply:
            names = [APP_WORDS[a][0] for a in apply]
            names = (
                ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
            )
            out.append(
                f"Overall: apply the changes for {names} ({','.join(apply)}); keep the rest."
            )
        else:
            out.append("Overall: keep the current settings.")
    return "\n".join(out).rstrip("\n")
