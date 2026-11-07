"""Which decisions to ask the user about, and the question for each, in plain words."""

from __future__ import annotations

from collections.abc import Callable

from decisions import BINARY, Decision, acts, features, level_of, tier_of


def distance(d: Decision, options: dict) -> tuple[float, bool] | None:
    """How far a decision is from the line that decides it, and which side it is on."""
    if d.app == "router":
        level = level_of(d)
        if level is None:
            return None
        cuts = (options["routerEasyBelow"], options["routerHardFrom"])
        nearest = min(cuts, key=lambda c: abs(level - c))
        return abs(level - nearest), level >= nearest
    values = features(d)
    if values is None:
        return None
    thresholds = tuple(options[k] for k in BINARY[d.app].keys)
    gaps = [abs(v - t) for v, t in zip(values, thresholds, strict=False)]
    return min(gaps), acts(values, thresholds)


def sample(
    decisions: list[Decision],
    labels: dict[str, object],
    options: dict,
    apps: list[str],
    per_side: int,
) -> list[Decision]:
    """For each app, the unlabeled decisions closest to its line, `per_side` on each side."""
    picked = []
    for app in apps:
        scored = [
            (gap, side, d)
            for d in decisions
            if d.app == app and d.id not in labels
            for gap, side in [distance(d, options) or (None, None)]
            if gap is not None
        ]
        for side in (True, False):
            near = sorted((x for x in scored if x[1] == side), key=lambda x: x[0])[
                :per_side
            ]
            picked.extend(d for _, _, d in near)
    return picked


def _clip(text: str, width: int = 110) -> str:
    one = " ".join(text.split())
    return one if len(one) <= width else one[: width - 1] + "…"


def _option(label: str, description: str, value: str) -> dict:
    return {"label": label, "description": description, "value": value}


def _yes_no(yes: tuple[str, str], no: tuple[str, str]) -> list[dict]:
    return [_option(*yes, "yes"), _option(*no, "no")]


def _score(d: Decision) -> str:
    return "-" if d.score is None else f"{d.score:.2f}"


def _router(d: Decision, subject: str) -> tuple[str, list[dict]]:
    level = level_of(d)
    score = "-" if level is None else f"{level:.2f}"
    model = str(d.detail.get("model") or "the default model")
    if d.verdict == "kept" or tier_of(model) is None:
        return (
            f'For the subagent task "{subject}", vSR kept {model} because the classifier was unsure (difficulty {score} of 2). Which model would have fit?',
            [
                _option("Haiku", "a quick lookup or formatting job", "haiku"),
                _option("Sonnet", "a few steps of ordinary work", "sonnet"),
                _option("Opus", "hard reasoning or deep expertise", "opus"),
            ],
        )
    return (
        f'For the subagent task "{subject}", vSR sent it to {model} (difficulty {score} of 2). Was that model right?',
        [
            _option("Too low", "it needed a stronger model", "too-low"),
            _option("Right", "that model fit the task", "right"),
            _option("Too high", "a lighter model would have done", "too-high"),
        ],
    )


def _guard(d: Decision, subject: str) -> tuple[str, list[dict]]:
    did = (
        "held it for you"
        + (", and you declined" if d.detail.get("outcome") == "declined" else "")
        if d.verdict == "held"
        else "let it run"
    )
    return (
        f"Should vSR have asked you before `{subject}`? (risk score {_score(d)}; vSR {did})",
        _yes_no(
            ("Ask first", "this command deserves a confirmation"),
            ("Let it run", "routine enough to run without asking"),
        ),
    )


def _escalate(d: Decision, subject: str) -> tuple[str, list[dict]]:
    did = (
        "sent the next subagent to Opus"
        if d.verdict == "escalated"
        else "took it as ordinary"
    )
    values = features(d) or (d.score or 0.0, 0.0)
    scores = f"pushback {values[0]:.2f}" + (
        f", repeat {values[1]:.2f}" if values[1] else ""
    )
    return (
        f'Was "{subject}" a real complaint about Claude\'s last answer? ({scores}; vSR {did})',
        _yes_no(
            ("A complaint", "the last answer was wrong or not what you wanted"),
            ("Not a complaint", "a new request, a thank-you or a question"),
        ),
    )


def _redact(d: Decision, subject: str) -> tuple[str, list[dict]]:
    counts = d.detail.get("counts") if isinstance(d.detail.get("counts"), dict) else {}
    what = ", ".join(f"{tag} ({n})" for tag, n in counts.items()) or "some text"
    source = d.detail.get("source") or subject
    return (
        f"vSR hid {what} in `{_clip(str(source))}` (score {_score(d)}). Was that personal data or a secret worth hiding?",
        _yes_no(
            ("Keep hiding", "it was personal data or a secret"),
            ("Fine to show", "it was ordinary text"),
        ),
    )


def _shield(d: Decision, subject: str) -> tuple[str, list[dict]]:
    did = "warned Claude" if d.verdict == "shielded" else "let it through"
    source = _clip(str(d.detail.get("source") or subject))
    return (
        f"Did `{source}` really try to give Claude instructions? (score {_score(d)}; vSR {did})",
        _yes_no(
            ("Yes, warn", "it tried to steer Claude"),
            ("No", "ordinary file or page content"),
        ),
    )


def _check(d: Decision, subject: str) -> tuple[str, list[dict]]:
    did = (
        "marked it unverified" if d.verdict == "unverified" else "passed it as grounded"
    )
    reads = (
        f"{d.detail.get('files', 0)} files and {d.detail.get('commands', 0)} commands"
    )
    return (
        f'Did Claude\'s answer to "{subject}" claim things that what it read ({reads}) does not back? (score {_score(d)}; vSR {did})',
        _yes_no(
            ("Unsupported", "it made something up or went beyond what it read"),
            ("Backed", "what it said matches what it read"),
        ),
    )


# Each app's question and answer options, for one of its decisions.
QUESTIONS: dict[str, Callable[[Decision, str], tuple[str, list[dict]]]] = {
    "router": _router,
    "guard": _guard,
    "escalate": _escalate,
    "redact": _redact,
    "shield": _shield,
    "check": _check,
}


def question(d: Decision) -> dict:
    """An AskUserQuestion entry for one decision, in plain words, with the label each option saves."""
    text, options = QUESTIONS[d.app](d, _clip(d.subject))
    return {
        "id": d.id,
        "app": d.app,
        "header": d.app.capitalize(),
        "question": text,
        "options": options,
    }
