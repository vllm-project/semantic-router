"""Paired CIs on the same held-out items. Numpy-only; GPU dumps stay off git."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def kv_fit_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError(
            f"KV tensors must have matching nonempty shapes, got {pred.shape} vs {true.shape}"
        )
    pred, true = pred.reshape(-1), true.reshape(-1)
    err = pred - true
    rel = float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-8))
    cosine = float(
        np.dot(pred, true) / (np.linalg.norm(pred) * np.linalg.norm(true) + 1e-8)
    )
    ss_res = float(np.square(err).sum())
    ss_tot = float(np.square(true - true.mean()).sum())
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    return {"rel_err": rel, "cosine": cosine, "r2": r2}


def paired_deltas(arm: np.ndarray, reference: np.ndarray) -> np.ndarray:
    arm = np.asarray(arm, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if arm.shape != reference.shape:
        raise ValueError(
            f"paired scores must share one length, got {arm.shape} vs {reference.shape}"
        )
    if arm.size == 0:
        raise ValueError("paired scores are empty")
    return arm - reference


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1.0 - alpha / 2])
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
    }


def paired_contrast(
    arm: np.ndarray,
    reference: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    return bootstrap_mean_ci(paired_deltas(arm, reference), n_boot=n_boot, seed=seed)


def build_report(
    metric: str,
    arms: dict[str, list[dict[str, Any]]],
    *,
    reference: str = "cold",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if reference not in arms:
        raise ValueError(f"reference arm {reference!r} missing from {sorted(arms)}")
    def indexed(name: str, items: list[dict[str, Any]]) -> dict[str, float]:
        if not isinstance(items, list) or not items:
            raise ValueError(f"arm {name!r} needs nonempty records with id and score")
        result = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"] or "score" not in item:
                raise ValueError(f"arm {name!r} needs records with nonempty string id and score")
            if item["id"] in result:
                raise ValueError(f"duplicate example id {item['id']!r} in arm {name!r}")
            score = float(item["score"])
            if not np.isfinite(score):
                raise ValueError(f"nonfinite score in arm {name!r} for {item['id']!r}")
            result[item["id"]] = score
        return result

    by_arm = {name: indexed(name, items) for name, items in arms.items()}
    ids = list(by_arm[reference])
    for name, scores in by_arm.items():
        if scores.keys() != by_arm[reference].keys():
            raise ValueError(f"arm {name!r} example IDs differ from {reference!r}")
    ref = np.asarray([by_arm[reference][item_id] for item_id in ids], dtype=np.float64)
    n = len(ids)
    contrasts = {}
    for name, scores in by_arm.items():
        if name == reference:
            continue
        contrasts[name] = paired_contrast(
            np.asarray([scores[item_id] for item_id in ids], dtype=np.float64), ref, n_boot=n_boot, seed=seed
        )
    return {
        "metric": metric,
        "reference": reference,
        "n_items": int(n),
        "example_ids": ids,
        "arm_means": {
            name: float(np.mean(list(scores.values())))
            for name, scores in by_arm.items()
        },
        "delta_vs_reference": contrasts,
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def read_items(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
