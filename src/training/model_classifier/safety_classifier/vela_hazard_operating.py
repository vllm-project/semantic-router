"""Development-only operating points constrained by safe any-label false alarms."""

import math

from .vela_hazard import score_predictions


def select_operating_point(
    rows, probabilities, labels, *, false_positive_budget=0.05, minimum_support=1
):
    """Fit one shared threshold; unobserved labels never count as known positives."""
    # Data-only tooling can import this module without numerical dependencies.
    import numpy as np  # noqa: PLC0415

    if (
        not math.isfinite(false_positive_budget)
        or not 0 <= false_positive_budget < 1
        or minimum_support < 1
    ):
        raise ValueError("Invalid false-positive budget or selection support")
    # Reuse the public shape/range contract before fitting any development point.
    baseline = score_predictions(rows, probabilities, labels)
    values = np.asarray(probabilities)
    targets = np.asarray([row["targets"] for row in rows], dtype=bool)
    masks = np.asarray([row["label_mask"] for row in rows], dtype=bool)
    safe = np.asarray([row.get("label") == "safe" for row in rows])
    if not safe.any() or targets[safe].any() or not masks[safe].all():
        raise ValueError("FP-constrained selection requires fully observed safe rows")
    eligible = [
        index
        for index, label in enumerate(labels)
        if min(
            baseline["per_label"][label]["positive_support"],
            baseline["per_label"][label]["negative_support"],
        )
        >= minimum_support
    ]
    if not eligible:
        raise ValueError("No development category has sufficient selection support")
    known_positive = (targets & masks).any(axis=1)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], values.ravel())))
    maximum_fp = math.floor(false_positive_budget * int(safe.sum()) + 1e-12)
    best = None
    for threshold in candidates:
        selected = values >= threshold
        false_alarms = int(selected[safe].any(axis=1).sum())
        if false_alarms > maximum_fp:
            continue
        tp = (selected & targets & masks).sum(axis=0)
        fp = (selected & ~targets & masks).sum(axis=0)
        fn = (~selected & targets & masks).sum(axis=0)
        denominator = 2 * tp + fp + fn
        f1 = np.divide(
            2 * tp,
            denominator,
            out=np.zeros(len(labels), dtype=float),
            where=denominator > 0,
        )
        correct_hit = float(
            (selected & targets & masks).any(axis=1)[known_positive].mean()
        )
        key = (float(f1[eligible].mean()), correct_hit, -false_alarms, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), false_alarms)
    result = {
        "feasible": best is not None,
        "selection_score": best[0][0] if best else -1.0,
        "false_positive_budget": false_positive_budget,
        "maximum_safe_false_positives": maximum_fp,
        "safe_support": int(safe.sum()),
        "selection_labels": [labels[index] for index in eligible],
        "minimum_support": minimum_support,
        "test_used": False,
        "protocol": (
            "One shared threshold fits supported-category macro-F1 under empirical "
            "safe any-label FP budget. All output labels count toward false alarms. "
            "Ties prefer correct known-positive category hits, fewer safe false "
            "alarms, then higher threshold. This is not posterior calibration "
            "or a population false-positive guarantee."
        ),
    }
    if best:
        result.update(
            {
                "threshold": best[1],
                "safe_false_positives": best[2],
                "known_positive_category_hit_recall": best[0][1],
                "metrics_all_labels": score_predictions(
                    rows, values, labels, [best[1]] * len(labels)
                ),
            }
        )
    return result
