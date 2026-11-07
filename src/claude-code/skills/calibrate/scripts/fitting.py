"""The fitting maths: thresholds that best agree with the user's labels, each moved at most
MAX_STEP per run.
"""

from __future__ import annotations

from dataclasses import dataclass

from decisions import SHIFTS, TIERS, Decision, acts, as_number, level_of, tier_of

MIN_LABELS = 8
MAX_STEP = 0.15
GRID = 0.01
# Each calibration goal's F-beta: below 1 favours fewer false alarms, above 1 catching more.
GOALS = {"balanced": 1.0, "fewer-false-alarms": 0.5, "catch-more": 2.0}


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
        if level is None or d.id not in labels:
            continue
        label = str(labels[d.id])
        tier = tier_of(d.detail.get("model"))
        if label in TIERS:
            target = TIERS.index(label)
        elif label in SHIFTS and tier is not None:
            target = min(2, max(0, tier + SHIFTS[label]))
        else:
            continue
        confidence = as_number(d.detail.get("confidence"))
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
