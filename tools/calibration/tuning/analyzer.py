"""OfflineAnalyzer — analytical threshold optimization on cached trace data.

Applies the same threshold-fix and regression-check logic as the live
pipeline, but operates entirely on pre-collected data without querying
the router.  Useful for confidence-based routing optimization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThresholdFix:
    """Regression-checked threshold fix for a single category/domain."""

    category: str
    old_threshold: float
    new_threshold: float
    affected: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    severity_gain: int = 0
    severity_loss: int = 0
    net_improvement: int = 0


class OfflineAnalyzer:
    """Analytical threshold optimization using cached traces (no live router).

    Constructor args:
        severity_fn: (item) -> int, severity weight for loss function.
        epsilon: margin added to candidate thresholds.
    """

    def __init__(self, severity_fn: Callable[[Any], int], epsilon: float = 0.001):
        self.severity_fn = severity_fn
        self.epsilon = epsilon

    def compute_threshold_fix(
        self,
        items: list[Any],
        current_threshold: float,
        target_threshold: float,
        confidence_fn: Callable[[Any], float],
        quadrant_fn: Callable[[Any], str],
        id_fn: Callable[[Any], str],
    ) -> ThresholdFix:
        """Analytical regression check for a threshold change.

        Escalation rule: escalate if confidence < threshold.
        Only items with confidence in [min(τ,τ'), max(τ,τ')] can change outcome.
        """
        lo = min(current_threshold, target_threshold)
        hi = max(current_threshold, target_threshold)

        affected, regressions = [], []
        severity_gain, severity_loss = 0, 0

        for item in items:
            conf = confidence_fn(item)
            if lo <= conf <= hi:
                item_id = id_fn(item)
                affected.append(item_id)
                w = self.severity_fn(item)

                old_esc = conf < current_threshold
                new_esc = conf < target_threshold
                if old_esc == new_esc:
                    continue

                quad = quadrant_fn(item)
                if new_esc:
                    if quad == "uplift":
                        severity_gain += w
                    elif quad == "regression":
                        severity_loss += w
                        regressions.append(item_id)
                elif quad == "uplift":
                    severity_loss += w
                    regressions.append(item_id)
                elif quad == "regression":
                    severity_gain += w

        return ThresholdFix(
            category="",
            old_threshold=current_threshold,
            new_threshold=target_threshold,
            affected=affected,
            regressions=regressions,
            severity_gain=severity_gain,
            severity_loss=severity_loss,
            net_improvement=severity_gain - severity_loss,
        )

    def find_optimal_threshold(
        self,
        items: list[Any],
        confidence_fn: Callable[[Any], float],
        quadrant_fn: Callable[[Any], str],
        correct_small_fn: Callable[[Any], bool],
        correct_large_fn: Callable[[Any], bool],
        id_fn: Callable[[Any], str],
    ) -> dict:
        """Sweep candidate thresholds and select the one maximizing accuracy.

        Returns dict with strategy classification (ESCALATE/SELECTIVE/AVOID),
        optimal threshold, accuracy, escalation rate, etc.
        """
        if not items:
            return {"strategy": "AVOID", "threshold": 0.0, "net": 0}

        conf_values = sorted({confidence_fn(q) for q in items})
        candidates = [0.0] + [c + self.epsilon for c in conf_values] + [1.001]

        best_threshold = 0.0
        best_accuracy = sum(1 for q in items if correct_small_fn(q))
        best_esc_rate = 0.0
        best_net = 0
        all_candidates = []

        for tau in candidates:
            escalated = [q for q in items if confidence_fn(q) < tau]
            kept = [q for q in items if confidence_fn(q) >= tau]

            correct = sum(1 for q in kept if correct_small_fn(q)) + sum(
                1 for q in escalated if correct_large_fn(q)
            )
            uplifts = sum(1 for q in escalated if quadrant_fn(q) == "uplift")
            regressions = sum(1 for q in escalated if quadrant_fn(q) == "regression")
            net = uplifts - regressions
            esc_rate = len(escalated) / len(items) if items else 0.0

            fix = self.compute_threshold_fix(
                items, 0.0, tau, confidence_fn, quadrant_fn, id_fn
            )

            all_candidates.append(
                {
                    "threshold": round(tau, 4),
                    "correct": correct,
                    "accuracy": round(100 * correct / len(items), 1),
                    "escalation_rate": round(100 * esc_rate, 1),
                    "uplifts": uplifts,
                    "regressions": regressions,
                    "net": net,
                    "severity_net": fix.net_improvement,
                }
            )

            if correct > best_accuracy or (
                correct == best_accuracy and len(escalated) < best_esc_rate * len(items)
            ):
                best_accuracy = correct
                best_threshold = tau
                best_net = net
                best_esc_rate = esc_rate

        escalate_ceiling = 0.85
        if best_net <= 0:
            strategy = "AVOID"
        elif best_esc_rate > escalate_ceiling:
            strategy = "ESCALATE"
        else:
            strategy = "SELECTIVE"

        baseline = sum(1 for q in items if correct_small_fn(q))
        return {
            "strategy": strategy,
            "optimal_threshold": round(best_threshold, 4),
            "best_accuracy": best_accuracy,
            "best_accuracy_pct": round(100 * best_accuracy / len(items), 1),
            "baseline_accuracy": baseline,
            "baseline_accuracy_pct": round(100 * baseline / len(items), 1),
            "escalation_rate": round(100 * best_esc_rate, 1),
            "net_uplift": best_net,
            "n_items": len(items),
            "candidates": all_candidates,
        }
