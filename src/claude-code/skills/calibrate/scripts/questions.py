"""Which decisions to ask the user about, and the question for each, in plain words."""

from __future__ import annotations

from collections.abc import Callable

import plain
from decisions import BINARY, TIERS, Decision, acts, features, level_of, tier_of


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


def _router(d: Decision, subject: str, options: dict) -> tuple[str, list[dict]]:
    level = level_of(d)
    judged = (
        ""
        if level is None
        else plain.difficulty(
            level, options["routerEasyBelow"], options["routerHardFrom"]
        )
    )
    task = f'A subagent was given the task "{subject}".'
    tier = tier_of(d.detail.get("model"))
    if d.verdict == "kept" or tier is None:
        unsure = (
            f"vSR judged it {judged} but was not sure" if judged else "vSR was not sure"
        )
        return (
            f"{task} {unsure} enough, so the subagent kept {plain.KEPT}. Which model would have fit?",
            [
                _option("Haiku", "a quick lookup or formatting job", "haiku"),
                _option("Sonnet", "a few steps of ordinary work", "sonnet"),
                _option("Opus", "hard reasoning or deep expertise", "opus"),
            ],
        )
    model = plain.model_name(d.detail.get("model"))
    judged = f" judged it {judged} and" if judged else ""
    answers = [_option("Right model", f"{model} fit the task", "right")]
    if tier < len(TIERS) - 1:
        answers.append(
            _option(
                "Needed stronger",
                f"a stronger model than {model} would have done better",
                "too-low",
            )
        )
    if tier > 0:
        answers.append(
            _option(
                "Could be lighter",
                f"a lighter model than {model} would have done",
                "too-high",
            )
        )
    return f"{task} vSR{judged} sent it to {model}. Was that the right model?", answers


def _guard(d: Decision, subject: str, _options: dict) -> tuple[str, list[dict]]:
    if d.verdict == "held":
        did = "held this command for you" + (
            ", and you declined it" if d.detail.get("outcome") == "declined" else ""
        )
    else:
        did = "let this command run without asking"
    return (
        f"vSR {did}: `{subject}`. Should vSR ask you before a command like this?",
        _yes_no(
            ("Ask first", "this command deserves a confirmation"),
            ("Let it run", "routine enough to run without asking"),
        ),
    )


def _escalate(d: Decision, subject: str, _options: dict) -> tuple[str, list[dict]]:
    did = (
        "took it as a complaint and sent the next subagent to Opus"
        if d.verdict == "escalated"
        else "took it as ordinary"
    )
    return (
        f'You wrote "{subject}", and vSR {did}. Was it a complaint about Claude\'s last answer?',
        _yes_no(
            ("A complaint", "the last answer was wrong or not what you wanted"),
            ("Not a complaint", "a new request, a thank-you or a question"),
        ),
    )


def _redact(d: Decision, subject: str, _options: dict) -> tuple[str, list[dict]]:
    counts = d.detail.get("counts") if isinstance(d.detail.get("counts"), dict) else {}
    what = ", ".join(f"{tag} ({n})" for tag, n in counts.items()) or "some text"
    source = d.detail.get("source") or subject
    return (
        f"vSR hid {what} in `{_clip(str(source))}` before Claude read it. Was that personal data or a secret worth hiding?",
        _yes_no(
            ("Keep hiding", "it was personal data or a secret"),
            ("Fine to show", "it was ordinary text"),
        ),
    )


def _shield(d: Decision, subject: str, _options: dict) -> tuple[str, list[dict]]:
    did = (
        "warned Claude that it may be trying to give instructions"
        if d.verdict == "shielded"
        else "let it through as ordinary content"
    )
    source = _clip(str(d.detail.get("source") or subject))
    return (
        f"vSR {did}: `{source}`. Did it really try to give Claude instructions?",
        _yes_no(
            ("Yes, warn", "it tried to steer Claude"),
            ("No", "ordinary file or page content"),
        ),
    )


def _check(d: Decision, subject: str, _options: dict) -> tuple[str, list[dict]]:
    did = "marked it unverified" if d.verdict == "unverified" else "passed it as backed"
    reads = (
        f"{d.detail.get('files', 0)} files and {d.detail.get('commands', 0)} commands"
    )
    return (
        f'For Claude\'s answer to "{subject}", vSR {did}. Did the answer claim things that what Claude read ({reads}) does not back?',
        _yes_no(
            ("Unsupported", "it made something up or went beyond what it read"),
            ("Backed", "what it said matches what it read"),
        ),
    )


# Each app's question and answer options, for one of its decisions.
QUESTIONS: dict[str, Callable[[Decision, str, dict], tuple[str, list[dict]]]] = {
    "router": _router,
    "guard": _guard,
    "escalate": _escalate,
    "redact": _redact,
    "shield": _shield,
    "check": _check,
}


def question(d: Decision, options: dict) -> dict:
    """An AskUserQuestion entry for one decision, in plain words, with the label each option
    saves; `options` are the thresholds in use, for the router's difficulty in words.
    """
    text, answers = QUESTIONS[d.app](d, _clip(d.subject), options)
    return {
        "id": d.id,
        "app": d.app,
        "header": plain.APP_WORDS[d.app][2],
        "question": text,
        "options": answers,
    }
