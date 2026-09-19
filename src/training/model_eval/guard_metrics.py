"""The metric contract a guard-type signal is scored under.

`mom_collection_eval.py` reports pooled accuracy and F1, which answers a
question a guard is not asked. A guard runs on every request, the router
compares its score with a configured threshold rather than taking an argmax,
and the cost of a mistake is different in each direction. Three measurements
reported on #3787 say what a pooled number hides on the shipped artifact: 339
of the 413 positives in the test split of its own training set carry no attack
marker, pooled recall of 0.4334 on that split covers per-band recall running
from 0.194 to 0.769, and a benign false-positive rate only means something next
to the budget it was allowed to spend.

So this module reports what #3194 settled on: separation, recall at a stated
benign false-positive budget, the same macro-averaged over length bands so
length cannot carry the score, named slices that are never pooled away, and
calibration, because the router thresholds the score rather than ranking it.
Every number carries the rows it was measured on.

Separation and the budget come from `provenance.metrics`, so a number this
module reports and the same number in an evaluation manifest are the same
computation. Nothing here loads a model, so a report can be recomputed from
saved scores, including a public guard's scores or a router response log.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from provenance.metrics import calibration_metrics, recall_at_fpr, roc_auc

# Word-count bands, as measured and published on #3787. Kept identical so a
# number from that report and a number from this module compare directly.
BANDS = ((1, 2), (3, 5), (6, 10), (11, 25), (26, 60), (61, 150), (151, None))
# config/config.yaml ships prompt_guard at 0.5. A run that scores a different
# operating point passes it in and the report records which one it used.
DEFAULT_THRESHOLD = 0.5
DEFAULT_BUDGETS = (0.001, 0.01)


def token_windows(
    count: int, size: int, overlap: int, specials: int = 2
) -> list[tuple[int, int]]:
    """The content ranges a windowed scan reads.

    The router does not truncate a long document, it scans it in windows and
    keeps the riskiest one, so a score quoted against a threshold belongs to a
    window rather than to the whole text. `size` counts the special tokens the
    model adds, the content width is what is left after them, and a window
    starts every `width - overlap` tokens until the document is covered. This
    is the geometry `candle-binding/src/core/sequence_windows.rs` scans with.
    """
    width = size - specials
    if width <= 0 or overlap >= width:
        raise ValueError("a window needs more content width than overlap")
    stride = width - overlap
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(start + width, count)
        ranges.append((start, end))
        if end == count:
            break
        start += stride
    return ranges


def band_of(words: int) -> str:
    """Name the length band a row of ``words`` words belongs to."""
    for low, high in BANDS:
        if high is None:
            if words >= low:
                return f"{low}+"
        elif low <= words <= high:
            return f"{low}-{high}"
    return "0"


def slice_rates(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[Any],
    threshold: float,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
) -> dict[str, dict[str, Any]]:
    """Report every group separately.

    A group holding one class reports its rates and no separation, because an
    area under the curve needs both sides and a benign-only source still owes a
    false-positive rate.
    """
    report: dict[str, dict[str, Any]] = {}
    for name in sorted({str(group) for group in groups}):
        members = [index for index, group in enumerate(groups) if str(group) == name]
        sliced_labels = [labels[index] for index in members]
        sliced_scores = [scores[index] for index in members]
        report[name] = _rates(sliced_labels, sliced_scores, threshold, budgets)
    return report


def routing_agreement(
    labels: Sequence[int],
    baseline: Sequence[float],
    candidate: Sequence[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Whether two guards route the same request the same way.

    Equal accuracy with different errors still moves traffic, so the rows the
    two disagree on are counted separately and attributed to whichever guard
    was right on them.
    """
    blocked_baseline = [score >= threshold for score in baseline]
    blocked_candidate = [score >= threshold for score in candidate]
    truth = [label == 1 for label in labels]
    disagreement = [
        (first, second, correct)
        for first, second, correct in zip(
            blocked_baseline, blocked_candidate, truth, strict=True
        )
        if first != second
    ]
    candidate_right = sum(1 for _, second, correct in disagreement if second == correct)
    baseline_right = len(disagreement) - candidate_right
    return {
        "rows": len(labels),
        "agreement_rate": 1.0 - len(disagreement) / len(labels),
        "both_block": sum(
            1
            for first, second in zip(blocked_baseline, blocked_candidate, strict=True)
            if first and second
        ),
        "candidate_blocks_only": sum(
            1 for first, second, _ in disagreement if second and not first
        ),
        "baseline_blocks_only": sum(
            1 for first, second, _ in disagreement if first and not second
        ),
        "on_disagreement_candidate_right": candidate_right,
        "on_disagreement_baseline_right": baseline_right,
        "mcnemar_exact_p": (
            two_sided_sign_test(candidate_right, len(disagreement))
            if disagreement
            else None
        ),
    }


def two_sided_sign_test(successes: int, trials: int) -> float:
    """Two-sided exact binomial probability against a fair coin.

    This is the exact McNemar test over the rows two guards disagree on. It is
    written out rather than imported so the contract stays on the standard
    library, and at the pair counts an evaluation set produces the exact sum
    costs milliseconds.
    """
    if trials <= 0:
        raise ValueError("an exact test needs at least one trial")
    observed = math.comb(trials, successes)
    # Every outcome no more likely than the observed one, which is what makes
    # the two-sided version exact rather than a doubled tail.
    total = sum(
        weight
        for weight in (math.comb(trials, k) for k in range(trials + 1))
        if weight <= observed
    )
    return min(1.0, total / 2**trials)


def bootstrap_interval(
    labels: Sequence[int],
    scores: Sequence[float],
    statistic: Callable[[Sequence[int], Sequence[float]], float | None],
    resamples: int = 2000,
    seed: int = 0,
    level: float = 95.0,
) -> dict[str, Any]:
    """Percentile bootstrap over rows for any statistic of (labels, scores).

    This covers evaluation-set sampling noise only. Training-seed variance is a
    separate source and needs repeated training runs rather than resampling.
    """
    rng = random.Random(seed)
    rows = len(labels)
    draws = []
    for _ in range(resamples):
        picked = [rng.randrange(rows) for _ in range(rows)]
        drawn_labels = [labels[index] for index in picked]
        if min(drawn_labels) == max(drawn_labels):
            continue
        value = statistic(drawn_labels, [scores[index] for index in picked])
        if value is not None:
            draws.append(value)
    if not draws:
        return {"point": statistic(labels, scores), "interval": None}
    draws.sort()
    tail = (1.0 - level / 100.0) / 2.0
    return {
        "point": statistic(labels, scores),
        "interval": [_quantile(draws, tail), _quantile(draws, 1.0 - tail)],
        "level": level,
        "resamples": len(draws),
    }


def guard_report(
    labels: Sequence[int],
    scores: Sequence[float],
    words: Sequence[int] | None = None,
    slices: dict[str, Sequence[Any]] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    bootstrap_resamples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """The contract for one score vector.

    ``words`` gives the word count per row and drives the length bands; without
    it the band sections are left out rather than guessed. ``slices`` names the
    columns a number must be read on, the source it came from, the language it
    is written in, or the fixed regression rows a release has to keep passing.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must align")
    if not labels:
        raise ValueError("an evaluation needs at least one row")

    report: dict[str, Any] = {
        "threshold": threshold,
        "pooled": _rates(labels, scores, threshold, budgets)
        | {
            "note": (
                "a reference line only: sources that do not share a label "
                "definition cannot be pooled, so read the slice sections"
            )
        },
        # A guard is thresholded rather than ranked, so the distance between a
        # score and the frequency it stands for is the product. Predicting the
        # positive class on every row makes each bin's accuracy the positive
        # rate, which is what the score claims.
        "calibration": calibration_metrics(labels, [1] * len(labels), scores),
    }

    if words is not None:
        bands = [band_of(count) for count in words]
        report["by_band"] = slice_rates(labels, scores, bands, threshold, budgets)
        # Macro over the bands, so an artifact cannot win by being right where
        # the rows are. A band with only one class carries no recall and is
        # left out. A band where no threshold stays inside the budget counts as
        # zero rather than dropping out, because a guard that can only stay in
        # budget by flagging nothing catches nothing there.
        for budget in budgets:
            key = _budget_key(budget)
            reached = [
                (entry[key] or {}).get("recall", 0.0)
                for entry in report["by_band"].values()
                if key in entry
            ]
            report[f"band_macro_{key}"] = (
                sum(reached) / len(reached) if reached else None
            )

    if slices:
        report["by_slice"] = {
            name: slice_rates(labels, scores, groups, threshold, budgets)
            for name, groups in slices.items()
        }

    if bootstrap_resamples:
        report["intervals"] = {
            "auc": bootstrap_interval(
                labels, scores, _auc_or_none, bootstrap_resamples, seed
            ),
            _budget_key(budgets[-1]): bootstrap_interval(
                labels,
                scores,
                lambda drawn, drawn_scores: _recall_or_none(
                    drawn, drawn_scores, budgets[-1]
                ),
                bootstrap_resamples,
                seed,
            ),
        }
    return report


def _rates(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    budgets: Sequence[float],
) -> dict[str, Any]:
    positives = [label == 1 for label in labels]
    blocked = [score >= threshold for score in scores]
    positive_count = sum(positives)
    negative_count = len(labels) - positive_count
    entry: dict[str, Any] = {
        "rows": len(labels),
        "positives": positive_count,
        "block_rate": sum(blocked) / len(labels),
        "recall": (
            sum(
                1
                for flag, is_positive in zip(blocked, positives, strict=True)
                if flag and is_positive
            )
            / positive_count
            if positive_count
            else None
        ),
        "false_positive_rate": (
            sum(
                1
                for flag, is_positive in zip(blocked, positives, strict=True)
                if flag and not is_positive
            )
            / negative_count
            if negative_count
            else None
        ),
    }
    if positive_count and negative_count:
        entry["auc"] = roc_auc(scores, positives)
        for budget in budgets:
            entry[_budget_key(budget)] = recall_at_fpr(scores, positives, budget)
    return entry


def _budget_key(budget: float) -> str:
    return f"recall_at_{budget:g}_fpr"


def _auc_or_none(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [label == 1 for label in labels]
    if not any(positives) or all(positives):
        return None
    return roc_auc(scores, positives)


def _recall_or_none(
    labels: Sequence[int], scores: Sequence[float], budget: float
) -> float | None:
    """Recall inside the budget, or ``None`` when the draw cannot answer.

    ``recall_at_fpr`` reports nothing for two different reasons: the draw holds
    one class, which no recall is defined for, and the draw holds both but no
    threshold stays inside the budget. The second is a recall of zero, and
    dropping it would leave the bootstrap only the draws that reached the
    budget, which is every unfavourable outcome removed and an interval that
    cannot fall below the favourable ones.
    """
    positives = [label == 1 for label in labels]
    if not any(positives) or all(positives):
        return None
    reached = recall_at_fpr(scores, positives, budget)
    return reached["recall"] if reached else 0.0


def _quantile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
