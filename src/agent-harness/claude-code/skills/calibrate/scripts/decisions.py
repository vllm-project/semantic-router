"""vSR's decision log, read: decisions, the labels the user gave, the labels the log implies,
and the numbers each app compares with its thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

IMPLICIT_WINDOW_S = 15 * 60
TIERS = ("haiku", "sonnet", "opus")


@dataclass(frozen=True)
class Binary:
    """An app that acts when any of its features reaches its threshold."""

    keys: tuple[str, ...]
    noun: str


BINARY = {
    "guard": Binary(("guardReview",), "holds"),
    "escalate": Binary(("escalateFeedback", "escalateReask"), "escalations"),
    "redact": Binary(("redactPii",), "hides"),
    "shield": Binary(("shieldInject",), "warnings"),
    "check": Binary(("checkUnsupported",), "marks"),
}
ROUTER_KEYS = ("routerEasyBelow", "routerHardFrom", "routerMinConfidence")
APPS = ("router", *BINARY)
# The mod logs these only when they acted: the log cannot show what a lower threshold would add, so they only rise.
ACTED_ONLY = ("escalate", "redact", "shield")
BINARY_LABELS = {"yes": True, "no": False}
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
    # Whether the mod offers answers for it, so a label can move a threshold; a line logged
    # before the mod wrote this field is not learnable.
    learnable: bool = False


@dataclass
class Log:
    decisions: list[Decision]
    lines: int
    skipped: int


def as_number(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def parse_time(value: object) -> float | None:
    """Seconds since the epoch, from epoch seconds or milliseconds or an ISO 8601 string."""
    seconds = as_number(value)
    if seconds is not None:
        return seconds / 1000 if seconds > 1e11 else seconds
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
                score=as_number(raw.get("score")),
                threshold=as_number(raw.get("threshold")),
                subject=str(raw.get("subject") or ""),
                detail=raw.get("detail") if isinstance(raw.get("detail"), dict) else {},
                learnable=raw.get("learnable") is True,
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
            and d.learnable
        ):
            found[d.id] = "yes"
    routes = [
        d
        for d in decisions
        if d.app == "router"
        and d.verdict == "routed"
        and d.learnable
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
    """A learnable router decision's difficulty level: `score` holds it for a route, confidence
    for a keep.
    """
    if not d.learnable:
        return None
    level = as_number(d.detail.get("level"))
    return level if level is not None else (d.score if d.verdict == "routed" else None)


def features(d: Decision) -> tuple[float, ...] | None:
    """The numbers a learnable binary decision compared with its thresholds, in their order."""
    if not d.learnable or d.score is None:
        return None
    if d.app == "escalate":
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
                as_number(feedback.get("WRONG_ANSWER")),
                as_number(feedback.get("WANT_DIFFERENT")),
            )
            if v is not None
        ]
        return max(pushback, default=0.0), as_number(d.detail.get("reask")) or 0.0
    return (d.score,)


def acts(values: tuple[float, ...], thresholds: tuple[float, ...]) -> bool:
    return any(v >= t for v, t in zip(values, thresholds, strict=False))
