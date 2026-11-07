#!/usr/bin/env python3
"""Calibrate vSR's thresholds from the user's own decision log and labels.

Subcommands: report, sample, label, suggest, apply, probe. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "setup" / "scripts"))
import runtime
import settings

MIN_LABELS = 8
MAX_STEP = 0.15
GRID = 0.01
IMPLICIT_WINDOW_S = 15 * 60
TIERS = ("haiku", "sonnet", "opus")
GOALS = {"balanced": 1.0, "fewer-false-alarms": 0.5, "catch-more": 2.0}


@dataclass(frozen=True)
class Binary:
    """An app that acts when any of its features reaches its threshold."""

    keys: tuple[str, ...]
    acts: str
    noun: str
    header: str


BINARY = {
    "guard": Binary(("guardReview",), "held", "holds", "Guard"),
    "escalate": Binary(
        ("escalateFeedback", "escalateReask"), "escalated", "escalations", "Escalate"
    ),
    "redact": Binary(("redactPii",), "hidden", "hides", "Redact"),
    "shield": Binary(("shieldInject",), "shielded", "warnings", "Shield"),
    "check": Binary(("checkUnsupported",), "unverified", "marks", "Check"),
}
ROUTER_KEYS = ("routerEasyBelow", "routerHardFrom", "routerMinConfidence")
APPS = ("router", *BINARY)
# The mod logs these only when they acted: the log cannot show what a lower threshold would add, so they only rise.
ACTED_ONLY = ("escalate", "redact", "shield")
BINARY_LABELS = {
    "yes": True,
    "no": False,
    True: True,
    False: False,
    "true": True,
    "false": False,
}
ROUTER_LABELS = ("too-low", "right", "too-high", *TIERS)
SHIFTS = {"too-low": 1, "right": 0, "too-high": -1}


@dataclass
class Decision:
    id: str
    at: float | None
    session: str
    app: str
    verdict: str
    score: float | None
    threshold: float | None
    subject: str
    detail: dict = field(default_factory=dict)


@dataclass
class Log:
    decisions: list[Decision]
    lines: int
    skipped: int


def _number(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def parse_time(value: object) -> float | None:
    """Seconds since the epoch, from epoch seconds or milliseconds or an ISO 8601 string."""
    number = _number(value)
    if number is not None:
        return number / 1000 if number > 1e11 else number
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def read_log(path: Path) -> Log:
    """The decisions in a JSONL log; a line that does not parse (a partial last write) is skipped and counted."""
    if not path.exists():
        return Log([], 0, 0)
    decisions, skipped, lines = [], 0, 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        lines += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if (
            not isinstance(raw, dict)
            or raw.get("app") not in APPS
            or not isinstance(raw.get("id"), str)
        ):
            skipped += 1
            continue
        decisions.append(
            Decision(
                id=raw["id"],
                at=parse_time(raw.get("at")),
                session=str(raw.get("session") or ""),
                app=raw["app"],
                verdict=str(raw.get("verdict") or ""),
                score=_number(raw.get("score")),
                threshold=_number(raw.get("threshold")),
                subject=str(raw.get("subject") or ""),
                detail=raw.get("detail") if isinstance(raw.get("detail"), dict) else {},
            )
        )
    return Log(decisions, lines, skipped)


def read_labels(path: Path) -> dict[str, object]:
    """id -> label value from labels.jsonl; the last label for an id wins."""
    labels: dict[str, object] = {}
    if not path.exists():
        return labels
    for line in path.read_text(errors="replace").splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict) and isinstance(raw.get("id"), str) and "label" in raw:
            labels[raw["id"]] = raw["label"]
    return labels


def tier_of(model: object) -> int | None:
    text = str(model or "").lower()
    return next((i for i, name in enumerate(TIERS) if name in text), None)


def implicit_labels(decisions: list[Decision]) -> dict[str, object]:
    """Labels the log implies: a hold the user declined, and a route followed soon by an escalation."""
    found: dict[str, object] = {}
    for d in decisions:
        if (
            d.app == "guard"
            and d.verdict == "held"
            and d.detail.get("outcome") == "declined"
            and features(d) is not None
        ):
            found[d.id] = "yes"
    routes = [
        d
        for d in decisions
        if d.app == "router"
        and d.verdict == "routed"
        and not d.detail.get("escalated")
        and d.at is not None
    ]
    for e in decisions:
        if e.app != "escalate" or e.verdict != "escalated" or e.at is None:
            continue
        before = [
            r
            for r in routes
            if r.session == e.session and 0 <= e.at - r.at <= IMPLICIT_WINDOW_S
        ]
        if before:
            last = max(before, key=lambda r: r.at)
            if tier_of(last.detail.get("model")) in (0, 1):
                found.setdefault(last.id, "too-low")
    return found


def level_of(d: Decision) -> float | None:
    """A router decision's difficulty level: `score` holds it for a route, confidence for a keep."""
    level = _number(d.detail.get("level"))
    return level if level is not None else (d.score if d.verdict == "routed" else None)


def features(d: Decision) -> tuple[float, ...] | None:
    """The numbers a binary app compares with its thresholds, in the order of its keys."""
    if d.app == "escalate":
        if d.score is None:
            return None
        if "feedback" not in d.detail and "reask" not in d.detail:
            return d.score, 0.0
        feedback = (
            d.detail.get("feedback")
            if isinstance(d.detail.get("feedback"), dict)
            else {}
        )
        pushback = [
            v
            for v in (
                _number(feedback.get("WRONG_ANSWER")),
                _number(feedback.get("WANT_DIFFERENT")),
            )
            if v is not None
        ]
        return max(pushback, default=0.0), _number(d.detail.get("reask")) or 0.0
    if d.app == "redact":
        counts = (
            d.detail.get("counts") if isinstance(d.detail.get("counts"), dict) else {}
        )
        if counts and set(counts) <= {"secret"}:
            return None
    if d.app == "guard" and (d.detail.get("unchecked") or d.detail.get("rule")):
        # Held for having more parts than the guard checks, or by a fixed rule, not for its score.
        return None
    return None if d.score is None else (d.score,)


def acts(values: tuple[float, ...], thresholds: tuple[float, ...]) -> bool:
    return any(v >= t for v, t in zip(values, thresholds, strict=False))


def fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
    b2 = beta * beta
    denominator = (1 + b2) * tp + b2 * fn + fp
    return 0.0 if denominator == 0 else (1 + b2) * tp / denominator


def axis(low: float, high: float) -> list[float]:
    return [round(low + i * GRID, 2) for i in range(round((high - low) / GRID) + 1)]


def grid(dimensions: int) -> list[tuple[float, ...]]:
    combos: list[tuple[float, ...]] = [()]
    for _ in range(dimensions):
        combos = [(*combo, value) for combo in combos for value in axis(0.05, 0.99)]
    return combos


def gap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return round(sum(abs(x - y) for x, y in zip(a, b, strict=False)), 6)


def best_toward(
    candidates: list[tuple[float, ...]], currents: tuple[float, ...], objective
) -> tuple[float, ...]:
    """The best candidate within MAX_STEP of `currents` on every axis.

    Best is the lowest `objective`; ties go to the candidate nearest the best one anywhere
    (the one nearest `currents`), so a run that cannot reach the best still moves toward it.
    """
    candidates = [*candidates, currents]
    anywhere = min(candidates, key=lambda c: (objective(c), gap(c, currents)))
    reachable = [
        c
        for c in candidates
        if all(abs(x - y) <= MAX_STEP + 1e-9 for x, y in zip(c, currents, strict=False))
    ]
    return min(reachable, key=lambda c: (objective(c), gap(c, anywhere)))


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def agree(self) -> int:
        return self.tp + self.tn


def confusion(
    cases: list[tuple[tuple[float, ...], bool]], thresholds: tuple[float, ...]
) -> Score:
    s = Score(0, 0, 0, 0)
    for values, truth in cases:
        predicted = acts(values, thresholds)
        if predicted and truth:
            s.tp += 1
        elif predicted:
            s.fp += 1
        elif truth:
            s.fn += 1
        else:
            s.tn += 1
    return s


def fit_binary(
    cases: list[tuple[tuple[float, ...], bool]],
    currents: tuple[float, ...],
    beta: float = 1.0,
    raise_only: bool = False,
) -> dict:
    """The thresholds that best separate the labels, moved at most MAX_STEP toward them.

    Best is the highest F-beta, then the fewest disagreements, then the smallest move.
    """
    if len(cases) < MIN_LABELS:
        return {
            "thresholds": currents,
            "changed": False,
            "why": f"{len(cases)} labels; needs {MIN_LABELS}",
        }

    def objective(candidate: tuple[float, ...]):
        m = confusion(cases, candidate)
        return (-round(fbeta(m.tp, m.fp, m.fn, beta), 9), m.fp + m.fn)

    candidates = grid(len(currents))
    if raise_only:
        candidates = [
            c
            for c in candidates
            if all(x >= y for x, y in zip(c, currents, strict=False))
        ]
    best = best_toward(candidates, currents, objective)
    changed = any(abs(b - c) > 1e-9 for b, c in zip(best, currents, strict=False))
    return {
        "thresholds": best,
        "changed": changed,
        "why": (
            "best separation of the labels"
            if changed
            else "the current thresholds already separate the labels best"
        ),
    }


def route_tier(level: float, easy: float, hard: float) -> int:
    return 0 if level < easy else (1 if level < hard else 2)


@dataclass(frozen=True)
class RouterCase:
    level: float
    confidence: float
    target: int


def router_cases(
    decisions: list[Decision], labels: dict[str, object]
) -> list[RouterCase]:
    cases = []
    for d in decisions:
        level = level_of(d) if d.app == "router" else None
        if level is None or d.id not in labels or d.detail.get("escalated"):
            continue
        label = str(labels[d.id])
        tier = tier_of(d.detail.get("model"))
        if label in TIERS:
            target = TIERS.index(label)
        elif label in SHIFTS and tier is not None:
            target = min(2, max(0, tier + SHIFTS[label]))
        else:
            continue
        confidence = _number(d.detail.get("confidence"))
        cases.append(
            RouterCase(level, 1.0 if confidence is None else confidence, target)
        )
    return cases


def fit_router(cases: list[RouterCase], easy: float, hard: float, least: float) -> dict:
    """Cuts that put the most labeled tasks in their right tier, then the confidence floor that routes
    only where routing is right more often than wrong; each moved at most MAX_STEP toward its best.
    """
    currents = (easy, hard, least)
    if len(cases) < MIN_LABELS:
        return {
            "thresholds": currents,
            "changed": False,
            "why": f"{len(cases)} labels; needs {MIN_LABELS}",
        }
    levels = axis(0.0, 2.0)
    cuts = [(e, h) for e in levels for h in levels if e < h]
    best_e, best_h = best_toward(
        cuts,
        (easy, hard),
        lambda c: -sum(route_tier(k.level, *c) == k.target for k in cases),
    )

    def net(m):
        return -sum(
            (1 if route_tier(k.level, best_e, best_h) == k.target else -1)
            for k in cases
            if k.confidence >= m[0]
        )

    (best_m,) = best_toward([(m,) for m in axis(0.0, 1.0)], (least,), net)
    best = (best_e, best_h, best_m)
    changed = any(abs(b - c) > 1e-9 for b, c in zip(best, currents, strict=False))
    return {
        "thresholds": best,
        "changed": changed,
        "why": (
            "more labeled tasks in their right tier"
            if changed
            else "the current cuts already fit the labels best"
        ),
    }


def router_agreement(
    cases: list[RouterCase], easy: float, hard: float, least: float
) -> tuple[int, int, int]:
    routed = [k for k in cases if k.confidence >= least]
    return (
        sum(route_tier(k.level, easy, hard) == k.target for k in routed),
        len(routed),
        len(cases) - len(routed),
    )


def distance(d: Decision, options: dict) -> tuple[float, bool] | None:
    """How far a decision is from the line that decides it, and which side it is on."""
    if d.app == "router":
        level = level_of(d)
        if level is None or d.detail.get("escalated"):
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


def question(d: Decision, options: dict) -> dict:
    """An AskUserQuestion entry for one decision, in plain words, with the label each option saves."""
    subject = _clip(d.subject)
    score = "-" if d.score is None else f"{d.score:.2f}"
    if d.app == "router":
        level = level_of(d)
        score = "-" if level is None else f"{level:.2f}"
        model = str(d.detail.get("model") or "the default model")
        if d.verdict == "kept" or tier_of(model) is None:
            return {
                "id": d.id,
                "app": "router",
                "header": "Router",
                "question": f'For the subagent task "{subject}", vSR kept {model} because the classifier was unsure (difficulty {score} of 2). Which model would have fit?',
                "options": [
                    {
                        "label": "Haiku",
                        "description": "a quick lookup or formatting job",
                        "value": "haiku",
                    },
                    {
                        "label": "Sonnet",
                        "description": "a few steps of ordinary work",
                        "value": "sonnet",
                    },
                    {
                        "label": "Opus",
                        "description": "hard reasoning or deep expertise",
                        "value": "opus",
                    },
                ],
            }
        did = f"sent it to {model}"
        return {
            "id": d.id,
            "app": "router",
            "header": "Router",
            "question": f'For the subagent task "{subject}", vSR {did} (difficulty {score} of 2). Was that model right?',
            "options": [
                {
                    "label": "Too low",
                    "description": "it needed a stronger model",
                    "value": "too-low",
                },
                {
                    "label": "Right",
                    "description": "that model fit the task",
                    "value": "right",
                },
                {
                    "label": "Too high",
                    "description": "a lighter model would have done",
                    "value": "too-high",
                },
            ],
        }

    def yes_no(yes, no):
        return [
            {"label": yes[0], "description": yes[1], "value": "yes"},
            {"label": no[0], "description": no[1], "value": "no"},
        ]

    if d.app == "guard":
        did = (
            "held it for you"
            + (", and you declined" if d.detail.get("outcome") == "declined" else "")
            if d.verdict == "held"
            else "let it run"
        )
        return {
            "id": d.id,
            "app": "guard",
            "header": "Guard",
            "question": f"Should vSR have asked you before `{subject}`? (risk score {score}; vSR {did})",
            "options": yes_no(
                ("Ask first", "this command deserves a confirmation"),
                ("Let it run", "routine enough to run without asking"),
            ),
        }
    if d.app == "escalate":
        did = (
            "sent the next subagent to Opus"
            if d.verdict == "escalated"
            else "took it as ordinary"
        )
        values = features(d) or (d.score or 0.0, 0.0)
        scores = f"pushback {values[0]:.2f}" + (
            f", repeat {values[1]:.2f}" if values[1] else ""
        )
        return {
            "id": d.id,
            "app": "escalate",
            "header": "Escalate",
            "question": f'Was "{subject}" a real complaint about Claude\'s last answer? ({scores}; vSR {did})',
            "options": yes_no(
                ("A complaint", "the last answer was wrong or not what you wanted"),
                ("Not a complaint", "a new request, a thank-you or a question"),
            ),
        }
    if d.app == "redact":
        counts = (
            d.detail.get("counts") if isinstance(d.detail.get("counts"), dict) else {}
        )
        what = ", ".join(f"{tag} ({n})" for tag, n in counts.items()) or "some text"
        source = d.detail.get("source") or subject
        return {
            "id": d.id,
            "app": "redact",
            "header": "Redact",
            "question": f"vSR hid {what} in `{_clip(str(source))}` (score {score}). Was that personal data or a secret worth hiding?",
            "options": yes_no(
                ("Keep hiding", "it was personal data or a secret"),
                ("Fine to show", "it was ordinary text"),
            ),
        }
    if d.app == "shield":
        did = "warned Claude" if d.verdict == "shielded" else "let it through"
        return {
            "id": d.id,
            "app": "shield",
            "header": "Shield",
            "question": f"Did `{_clip(str(d.detail.get('source') or subject))}` really try to give Claude instructions? (score {score}; vSR {did})",
            "options": yes_no(
                ("Yes, warn", "it tried to steer Claude"),
                ("No", "ordinary file or page content"),
            ),
        }
    did = (
        "marked it unverified" if d.verdict == "unverified" else "passed it as grounded"
    )
    reads = (
        f"{d.detail.get('files', 0)} files and {d.detail.get('commands', 0)} commands"
    )
    return {
        "id": d.id,
        "app": "check",
        "header": "Check",
        "question": f'Did Claude\'s answer to "{subject}" claim things that what it read ({reads}) does not back? (score {score}; vSR {did})',
        "options": yes_no(
            ("Unsupported", "it made something up or went beyond what it read"),
            ("Backed", "what it said matches what it read"),
        ),
    }


@dataclass
class Context:
    plugin_id: str
    options: dict
    log_path: Path
    log: Log
    explicit: dict[str, object]
    implicit: dict[str, object]

    @property
    def labels(self) -> dict[str, object]:
        return {**self.implicit, **self.explicit}

    @property
    def enabled(self) -> list[str]:
        return [app for app in APPS if self.options.get(app)]

    def path(self, name: str) -> Path:
        return self.log_path.with_name(name)


def context(args: argparse.Namespace) -> Context:
    plugin_id = args.id or settings.resolve_id(
        settings.listed_plugins(), args.plugin_root
    )
    options = settings.effective_options(
        settings.load(settings.settings_path())[1], plugin_id
    )
    log_path = Path(args.log or options["logPath"]).expanduser()
    log = read_log(log_path)
    explicit = read_labels(
        (args.labels and Path(args.labels)) or log_path.with_name("labels.jsonl")
    )
    return Context(
        plugin_id, options, log_path, log, explicit, implicit_labels(log.decisions)
    )


def binary_cases(ctx: Context, app: str) -> list[tuple[tuple[float, ...], bool]]:
    cases = []
    for d in ctx.log.decisions:
        if (
            d.app == app
            and d.id in ctx.labels
            and str(ctx.labels[d.id]).lower() in BINARY_LABELS
        ):
            values = features(d)
            if values is not None:
                cases.append((values, BINARY_LABELS[str(ctx.labels[d.id]).lower()]))
    return cases


def histogram(
    values: list[float], top: float, marks: list[float], bins: int = 10, width: int = 30
) -> list[str]:
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, max(0, int(v / top * bins)))] += 1
    biggest = max(counts) or 1
    lines = []
    for i, n in enumerate(counts):
        low, high = i * top / bins, (i + 1) * top / bins
        here = [m for m in marks if low <= m < high or (i == bins - 1 and m == top)]
        bar = "#" * round(n / biggest * width)
        lines.append(
            f"    {low:4.2f}-{high:4.2f} {bar:<{width}} {n:>4}"
            + (f"  <- {', '.join(f'{m:.2f}' for m in here)}" if here else "")
        )
    return lines


def report(ctx: Context) -> str:
    when = [d.at for d in ctx.log.decisions if d.at is not None]
    span = (
        f", {datetime.fromtimestamp(min(when)):%Y-%m-%d} to {datetime.fromtimestamp(max(when)):%Y-%m-%d}"
        if when
        else ""
    )
    out = [
        f"log {ctx.log_path}: {len(ctx.log.decisions)} decisions{span}"
        + (
            f"; {ctx.log.skipped} unreadable line(s) skipped" if ctx.log.skipped else ""
        ),
        f"settings {settings.settings_path()} under pluginConfigs[{ctx.plugin_id!r}]",
    ]
    last = last_calibration(ctx.path("calibrations.jsonl"))
    now = runtime_model(ctx.options["runtimeUrl"])
    out.append(
        f"runtime model now {now or 'unreachable'}; "
        + (
            f"last calibration {last.get('at')} with model {last.get('model')}"
            if last
            else "no calibration recorded yet"
        )
    )
    for app in APPS:
        mine = [d for d in ctx.log.decisions if d.app == app]
        off = sum(d.verdict == "off" for d in mine)
        state = "on" if ctx.options.get(app) else "off in settings"
        user = sum(1 for d in mine if d.id in ctx.explicit)
        implied = sum(
            1 for d in mine if d.id in ctx.implicit and d.id not in ctx.explicit
        )
        if app == "router":
            keys = ROUTER_KEYS
            values = [v for d in mine for v in [level_of(d)] if v is not None]
            top, near_width = 2.0, 0.2
        else:
            keys = BINARY[app].keys
            values = [f[0] for d in mine for f in [features(d)] if f is not None]
            top, near_width = 1.0, 0.1
        near = sum(
            1
            for d in mine
            for x in [distance(d, ctx.options)]
            if x is not None and x[0] <= near_width
        )
        lines = ", ".join(f"{k} {ctx.options[k]:.2f}" for k in keys)
        out.append(
            f"\n{app} ({state}): {len(mine)} decisions, {off} off, {near} near the line, labels {user} yours + {implied} implied · {lines}"
        )
        if app in ACTED_ONLY and mine:
            out.append(
                "    vSR logs this app only when it acted, so labels here can raise its threshold but not lower it; `probe` checks the other side"
            )
        if values:
            marks = [
                ctx.options[k]
                for k in keys
                if k not in {"routerMinConfidence", "escalateReask"}
            ]
            out.extend(histogram(values, top, marks))
        if app == "escalate":
            reasks = [
                f[1] for d in mine for f in [features(d)] if f is not None and f[1] > 0
            ]
            if reasks:
                out.append("    reask:")
                out.extend(histogram(reasks, 1.0, [ctx.options["escalateReask"]]))
    return "\n".join(out)


def suggestion(ctx: Context, goal: str) -> dict:
    beta = GOALS[goal]
    plan = {"pluginId": ctx.plugin_id, "goal": goal, "changes": {}, "apps": {}}
    for app in ctx.enabled:
        mine = [d for d in ctx.log.decisions if d.app == app]
        if not mine:
            plan["apps"][app] = {"why": "no decisions in the log"}
            continue
        if app == "router":
            cases = router_cases(ctx.log.decisions, ctx.labels)
            currents = tuple(ctx.options[k] for k in ROUTER_KEYS)
            fit = fit_router(cases, *currents)
            entry = {"labels": len(cases), "why": fit["why"], "before": {}, "after": {}}
            for name, values in (("before", currents), ("after", fit["thresholds"])):
                tiers = [0, 0, 0, 0]
                for d in mine:
                    level = level_of(d)
                    if level is None or d.detail.get("escalated"):
                        continue
                    confidence = _number(d.detail.get("confidence"))
                    tiers[
                        (
                            3
                            if confidence is not None and confidence < values[2]
                            else route_tier(level, values[0], values[1])
                        )
                    ] += 1
                right, routed, kept = router_agreement(cases, *values)
                entry[name] = {
                    "values": dict(zip(ROUTER_KEYS, values, strict=False)),
                    "haiku": tiers[0],
                    "sonnet": tiers[1],
                    "opus": tiers[2],
                    "kept": tiers[3],
                    "agree": f"{right}/{routed} routed right, {kept} kept",
                }
        else:
            spec = BINARY[app]
            cases = binary_cases(ctx, app)
            currents = tuple(ctx.options[k] for k in spec.keys)
            fit = fit_binary(cases, currents, beta, raise_only=app in ACTED_ONLY)
            entry = {"labels": len(cases), "why": fit["why"]}
            for name, values in (("before", currents), ("after", fit["thresholds"])):
                acted = sum(
                    1
                    for d in mine
                    for f in [features(d)]
                    if f is not None and acts(f, values)
                )
                s = confusion(cases, values)
                entry[name] = {
                    "values": dict(zip(spec.keys, values, strict=False)),
                    spec.noun: acted,
                    "agree": f"{s.agree}/{s.n}" if s.n else "-",
                    "falseAlarms": s.fp,
                    "missed": s.fn,
                }
        plan["apps"][app] = entry
        for k, before, after in zip(
            entry["before"]["values"],
            entry["before"]["values"].values(),
            entry["after"]["values"].values(),
            strict=False,
        ):
            if abs(after - before) > 1e-9:
                plan["changes"][k] = {"from": before, "to": after}
    return plan


def table(plan: dict) -> str:
    rows = [
        f"goal: {plan['goal']} · at most {MAX_STEP} per threshold per run · at least {MIN_LABELS} labels per app",
        "",
    ]
    rows.append(
        f"{'app':9} {'threshold':20} {'before':>7} {'after':>7}  {'on the log (before → after)':32} {'labels agree (before → after)':30}"
    )
    for app, entry in plan["apps"].items():
        if "before" not in entry:
            rows.append(f"{app:9} {entry['why']}")
            continue
        for i, (k, before) in enumerate(entry["before"]["values"].items()):
            after = entry["after"]["values"][k]
            mark = "" if abs(after - before) < 1e-9 else " *"
            if i == 0:
                if app == "router":
                    b, a = entry["before"], entry["after"]
                    acted = f"H/S/O/kept {b['haiku']}/{b['sonnet']}/{b['opus']}/{b['kept']} → {a['haiku']}/{a['sonnet']}/{a['opus']}/{a['kept']}"
                else:
                    noun = next(
                        n
                        for n in entry["before"]
                        if n not in ("values", "agree", "falseAlarms", "missed")
                    )
                    acted = f"{entry['before'][noun]} → {entry['after'][noun]} {noun}"
                agree = f"{entry['before']['agree']} → {entry['after']['agree']}"
            else:
                acted = agree = ""
            rows.append(
                f"{app if i == 0 else '':9} {k:20} {before:7.2f} {after:7.2f}{mark:2} {acted:32} {agree}"
            )
        rows.append(f"{'':9} {entry['labels']} labels: {entry['why']}")
    if not plan["changes"]:
        rows.append("\nno threshold changes suggested")
    return "\n".join(rows)


def settings_diff(plan: dict) -> str:
    path = settings.settings_path()
    text, current = settings.load(path)
    values = {k: v["to"] for k, v in plan["changes"].items()}
    return settings.diff(
        text, settings.render(settings.merge(current, plan["pluginId"], values)), path
    )


def runtime_model(url: str) -> str | None:
    return runtime.health(url, timeout=2)["model"]


def last_calibration(path: Path) -> dict | None:
    lines = path.read_text().splitlines() if path.exists() else []
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def apply(
    plan: dict,
    calibrations: Path,
    labels_by_app: dict[str, int],
    model: str | None = None,
) -> str:
    path = settings.settings_path()
    _text, current = settings.load(path)
    values = {k: settings.parse_value(k, v["to"]) for k, v in plan["changes"].items()}
    if not values:
        return "nothing to apply"
    backup = settings.write(
        path, settings.render(settings.merge(current, plan["pluginId"], values))
    )
    calibrations.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pluginId": plan["pluginId"],
        "goal": plan["goal"],
        "model": model,
        "changes": plan["changes"],
        "labels": labels_by_app,
    }
    with calibrations.open("a") as out:
        out.write(json.dumps(record) + "\n")
    return (
        f"wrote {path}"
        + (f" (previous copy: {backup})" if backup else "")
        + f"; recorded in {calibrations}"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id", help="plugin id the settings are keyed under (default: detected)"
    )
    parser.add_argument(
        "--plugin-root",
        help="the running copy's ${CLAUDE_PLUGIN_ROOT}, to detect the id",
    )
    parser.add_argument("--log", help="decision log (default: logPath from settings)")
    parser.add_argument(
        "--labels", help="labels file (default: labels.jsonl beside the log)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report")
    one = sub.add_parser("sample")
    one.add_argument("--per-side", type=int, default=2)
    one.add_argument("--apps", help="comma-separated apps (default: every enabled app)")
    one = sub.add_parser("label")
    one.add_argument("pairs", nargs="+", metavar="ID=LABEL")
    one = sub.add_parser("suggest")
    one.add_argument("--goal", choices=tuple(GOALS), default="balanced")
    one.add_argument(
        "--out",
        help="where to save the plan for `apply` (default: calibration-plan.json beside the log)",
    )
    one = sub.add_parser("apply")
    one.add_argument(
        "plan",
        nargs="?",
        help="the plan `suggest` saved (default: calibration-plan.json beside the log)",
    )
    one = sub.add_parser("probe")
    one.add_argument("--url", help="runtime URL (default: runtimeUrl from settings)")
    args = parser.parse_args(argv)
    try:
        ctx = context(args)
        if args.command == "report":
            print(report(ctx))
        elif args.command == "sample":
            apps = args.apps.split(",") if args.apps else ctx.enabled
            print(
                json.dumps(
                    [
                        question(d, ctx.options)
                        for d in sample(
                            ctx.log.decisions,
                            ctx.labels,
                            ctx.options,
                            apps,
                            args.per_side,
                        )
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.command == "label":
            by_id = {d.id: d for d in ctx.log.decisions}
            path = Path(args.labels) if args.labels else ctx.path("labels.jsonl")
            lines = []
            for pair in args.pairs:
                decision_id, _, value = pair.rpartition("=")
                if decision_id not in by_id:
                    raise settings.SettingsError(
                        f"no decision {decision_id!r} in {ctx.log_path}"
                    )
                app = by_id[decision_id].app
                allowed = ROUTER_LABELS if app == "router" else ("yes", "no")
                if value not in allowed:
                    raise settings.SettingsError(
                        f"{app} labels are {', '.join(allowed)}, not {value!r}"
                    )
                lines.append(
                    json.dumps({"id": decision_id, "app": app, "label": value}) + "\n"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as out:
                out.writelines(lines)
            print(f"saved {len(lines)} label(s) to {path}")
        elif args.command == "suggest":
            plan = suggestion(ctx, args.goal)
            print(table(plan))
            if plan["changes"]:
                print("\nsettings diff:\n" + settings_diff(plan), end="")
            out = Path(args.out) if args.out else ctx.path("calibration-plan.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(plan, indent=2) + "\n")
            print(f"\nplan saved to {out}")
        elif args.command == "apply":
            plan = json.loads(
                Path(args.plan or ctx.path("calibration-plan.json")).read_text()
            )
            labels = {
                app: entry.get("labels", 0) for app, entry in plan["apps"].items()
            }
            print(
                apply(
                    plan,
                    ctx.path("calibrations.jsonl"),
                    labels,
                    runtime_model(ctx.options["runtimeUrl"]),
                )
            )
        else:
            rows = runtime.run_probes(
                args.url or ctx.options["runtimeUrl"],
                runtime._read_token(None),
                ctx.options,
                set(ctx.enabled),
                smoke=False,
                tolerance=0.15,
            )
            runtime._print_probes(rows)
        return 0
    except settings.SettingsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
