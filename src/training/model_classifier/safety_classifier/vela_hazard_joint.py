"""Joint per-label development thresholds under explicit safe-union budgets."""

import math

from .vela_hazard import score_predictions

COORDINATE_UPDATES = 72
MINIMUM_F1_GAIN = 1e-12


def safe_group_indices(rows, safe_groups=None):
    """Resolve stable DEV IDs; always constrain all fully observed safe rows."""
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Development IDs must be unique")
    indices = {identity: index for index, identity in enumerate(ids)}
    safe = [index for index, row in enumerate(rows) if row.get("label") == "safe"]
    if not safe or any(
        any(rows[index]["targets"]) or not all(rows[index]["label_mask"])
        for index in safe
    ):
        raise ValueError("Joint selection requires fully observed safe rows")
    groups = {"all_safe": safe}
    if safe_groups is None:
        return groups
    if not isinstance(safe_groups, dict):
        raise ValueError("Safe groups must map names to development IDs")
    safe_set = set(safe)
    for name, members in safe_groups.items():
        if not isinstance(name, str) or not name or name == "all_safe":
            raise ValueError("Safe group names must be nonempty and not all_safe")
        if (
            not isinstance(members, list)
            or not members
            or any(not isinstance(member, str) for member in members)
            or len(set(members)) != len(members)
            or any(member not in indices for member in members)
        ):
            raise ValueError("Safe group members must be unique existing DEV IDs")
        group = [indices[member] for member in members]
        if any(index not in safe_set for index in group):
            raise ValueError("Safe groups may contain only fully observed safe rows")
        groups[name] = group
    return groups


def _candidates(values):
    import numpy as np  # noqa: PLC0415

    result = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
                values.ravel(),
                np.nextafter(values.ravel(), np.float32(np.inf)),
            )
        )
    )
    return result[(result >= 0) & (result <= 1)]


def _shared_start(rows, values, labels, limits):
    best = None
    for threshold in _candidates(values):
        selected = values >= threshold
        if any(selected[group].any(axis=1).sum() > limit for group, limit in limits):
            continue
        metrics = score_predictions(
            rows, values, labels, [float(threshold)] * len(labels)
        )
        key = (metrics["macro_f1"], -metrics["safe_any_hazard_rate"], float(threshold))
        if best is None or key > best[0]:
            best = key, float(threshold)
    return best[1] if best else None


def _coordinate_fit(rows, values, labels, limits, initial):
    import numpy as np  # noqa: PLC0415

    target = np.asarray([row["targets"] for row in rows], dtype=bool)
    mask = np.asarray([row["label_mask"] for row in rows], dtype=bool)
    safe = np.asarray([row.get("label") == "safe" for row in rows])
    thresholds = np.full(len(labels), initial, dtype=np.float32)
    selected = values >= thresholds
    if any(selected[group].any(axis=1).sum() > limit for group, limit in limits):
        raise ValueError("Saturated scores prevent conservative initialization")
    history = []
    for step in range(COORDINATE_UPDATES):
        before = score_predictions(rows, values, labels, thresholds)["macro_f1"]
        best = None
        for index, _label in enumerate(labels):
            others = selected.copy()
            others[:, index] = False
            other_any = others.any(axis=1)
            for threshold in _candidates(values[:, index]):
                proposed = values[:, index] >= threshold
                union = other_any | proposed
                if any(union[group].sum() > limit for group, limit in limits):
                    continue
                gold, known = target[:, index], mask[:, index]
                tp = (proposed & gold & known).sum()
                fp = (proposed & ~gold & known).sum()
                fn = (~proposed & gold & known).sum()
                old = selected[:, index]
                old_tp = (old & gold & known).sum()
                old_fp = (old & ~gold & known).sum()
                old_fn = (~old & gold & known).sum()
                new_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
                old_denominator = 2 * old_tp + old_fp + old_fn
                old_f1 = 2 * old_tp / old_denominator if old_denominator else 0
                gain = float(new_f1 - old_f1)
                key = (gain, -int(union[safe].sum()), float(threshold), -index)
                if gain > MINIMUM_F1_GAIN and (best is None or key > best[0]):
                    best = key, index, threshold
        if best is None:
            break
        _, index, threshold = best
        thresholds[index] = threshold
        selected[:, index] = values[:, index] >= threshold
        history.append(
            {
                "step": step,
                "label": labels[index],
                "threshold": float(threshold),
                "macro_f1_before": before,
                "macro_f1_after": score_predictions(rows, values, labels, thresholds)[
                    "macro_f1"
                ],
                "safe_false_positives": int(selected[safe].any(axis=1).sum()),
            }
        )
    return {
        "thresholds": thresholds.tolist(),
        "metrics": score_predictions(rows, values, labels, thresholds),
        "history": history,
    }


def select_joint_operating_point(
    rows,
    probabilities,
    labels,
    *,
    false_positive_budget=0.05,
    minimum_support=1,
    safe_groups=None,
):
    """Fit deterministic independent thresholds; preserve all-taxonomy macro F1."""
    import numpy as np  # noqa: PLC0415

    if (
        not math.isfinite(false_positive_budget)
        or not 0 <= false_positive_budget < 1
        or minimum_support != 1
    ):
        raise ValueError(
            "Joint selection requires a valid FP budget and minimum support 1"
        )
    # Validate before casting: rounding must not hide out-of-range input scores.
    score_predictions(rows, probabilities, labels)
    values = np.asarray(probabilities, dtype=np.float32)
    baseline = score_predictions(rows, values, labels)
    groups = safe_group_indices(rows, safe_groups)
    limits = [
        (
            np.asarray(indices, dtype=int),
            math.floor(false_positive_budget * len(indices) + 1e-12),
        )
        for indices in groups.values()
    ]
    shared = _shared_start(rows, values, labels, limits)
    result = {
        "feasible": shared is not None,
        "selection_score": -1.0,
        "false_positive_budget": false_positive_budget,
        "limits": [
            {"name": name, "safe_rows": len(indices), "max_fp": limit}
            for (name, indices), (_, limit) in zip(groups.items(), limits, strict=True)
        ],
        "minimum_positive_support": 1,
        "unsupported_or_rare_labels_not_certified": True,
        "unsupported_labels": [
            label
            for label, details in baseline["per_label"].items()
            if min(details["positive_support"], details["negative_support"]) < 1
        ],
        "selection_contract": (
            "Independent per-label FP32 thresholds under every safe-union budget. "
            "All taxonomy labels remain in macro F1 (unsupported F1 is zero) and "
            "all outputs count toward safe false alarms. Deterministic starts at "
            "1 and the best shared threshold; at most 72 improving coordinate steps. "
            "This is empirical development selection, not posterior calibration "
            "or a population false-positive guarantee."
        ),
        "test_used": False,
    }
    if shared is None:
        return result
    options = [
        _coordinate_fit(rows, values, labels, limits, initial)
        for initial in (1.0, shared)
    ]
    selected = max(
        options,
        key=lambda item: (
            item["metrics"]["macro_f1"],
            -item["metrics"]["safe_any_hazard_rate"],
        ),
    )
    result.update(selected)
    result["initializations"] = [1.0, shared]
    result["selection_score"] = selected["metrics"]["macro_f1"]
    return result
