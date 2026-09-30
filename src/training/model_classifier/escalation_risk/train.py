"""Train, calibrate, and evaluate the residual-failure escalation classifier (#3282).

Reads rows produced by fixtures.py (or, later, real rows with the same shape):

    train split        -> fit the model
    calibration split  -> fit probability calibration and pick the threshold
    test split         -> report held-out metrics against fixed baselines

Output is one JSON artifact (no pickle): feature schema, weights, calibration,
threshold, metrics, and a content digest as its identity. A model trained on
synthetic rows is TEST EVIDENCE ONLY.

Usage:
    python train.py --data synthetic.jsonl --out artifact.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ARTIFACT_VERSION = "escalation-risk-artifact-v1"
SEED = 0

# Feature schema. Every feature, whatever its type, also gets a 0/1
# "<name>:missing" column, so a missing value never looks like a real one:
#   categorical -> one-hot, plus missing flag
#   boolean     -> 0/1 value, plus missing flag (absent != confirmed false)
#   numeric     -> value, plus missing flag
# A status outside STATUSES, or a category outside the vocabulary, raises
# instead of being encoded as all zeros.
PRESENT = "present"
STATUSES = {PRESENT, "absent", "not_applicable"}
CATEGORICAL = {
    "decision": [
        "general_chat",
        "code_help",
        "math_reasoning",
        "legal_qa",
        "summarize",
    ],
    "primary_model": ["small-model-a", "small-model-b"],
    "prompt_tokens_bucket": ["<256", "256-1k", "1k-4k", "4k+"],
}
NUMERIC = [
    "complexity_score",
    "domain_confidence",
    "context_fill_ratio",
    "tool_count",
    "recent_no_progress_turns",
]
BOOLEAN = ["has_tools"]

# Target: catch at least this share of real failures (recall) on calibration data.
TARGET_RECALL = 0.80
LOW_CONFIDENCE_THRESHOLD = 0.6


# ---------- features ----------


def feature_names() -> list[str]:
    names: list[str] = []
    for key, values in CATEGORICAL.items():
        names += [f"{key}={v}" for v in values] + [f"{key}:missing"]
    for key in BOOLEAN + NUMERIC:
        names += [key, f"{key}:missing"]
    return names


def is_present(features: dict, key: str) -> bool:
    status = features[key]["status"]
    if status not in STATUSES:
        raise ValueError(f"feature {key!r} has unknown status {status!r}")
    return status == PRESENT


def encode(features: dict) -> list[float]:
    row: list[float] = []
    for key, values in CATEGORICAL.items():
        if not is_present(features, key):
            row += [0.0] * len(values) + [1.0]
            continue
        value = features[key]["value"]
        if value not in values:
            raise ValueError(f"feature {key!r} has unknown category {value!r}")
        row += [1.0 if value == option else 0.0 for option in values] + [0.0]
    for key in BOOLEAN:
        if is_present(features, key):
            row += [1.0 if features[key]["value"] else 0.0, 0.0]
        else:  # absent / not_applicable: never read as a confirmed False
            row += [0.0, 1.0]
    for key in NUMERIC:
        if is_present(features, key):
            row += [float(features[key]["value"]), 0.0]
        else:  # value slot 0, flag says it is missing rather than zero
            row += [0.0, 1.0]
    return row


def load(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict]]]:
    by_split: dict[str, list[dict]] = {}
    excluded = 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["label"] is None:  # tie / abstain / both_failed
            excluded += 1
            continue
        by_split.setdefault(r["split"], []).append(r)
    print(
        f"loaded: {sum(len(v) for v in by_split.values())} labelled, {excluded} excluded"
    )
    out = {}
    for split, rows in by_split.items():
        x = np.array([encode(r["features"]) for r in rows])
        y = np.array([r["label"] for r in rows])
        out[split] = (x, y, rows)
    return out


# ---------- calibration (Platt scaling) ----------


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(random_state=SEED).fit(logit(raw).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(raw: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * logit(raw) + b)))


# ---------- metrics ----------


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: how far 'predicted %' is from 'actual %'."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def decision_metrics(escalate: np.ndarray, y: np.ndarray) -> dict:
    pos, neg = y == 1, y == 0
    return {
        "false_negative_rate": float((~escalate & pos).sum() / max(1, pos.sum())),
        "false_positive_rate": float((escalate & neg).sum() / max(1, neg.sum())),
        "escalation_rate": float(escalate.mean()),
    }


def pick_threshold(p: np.ndarray, y: np.ndarray, target_recall: float) -> float:
    """Highest threshold that still catches target_recall of failures."""
    for t in np.round(np.arange(0.99, 0.0, -0.01), 2):
        if ((p >= t) & (y == 1)).sum() / max(1, (y == 1).sum()) >= target_recall:
            return float(t)
    return 0.0


# ---------- main ----------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifact.json"))
    args = ap.parse_args()

    data = load(args.data)
    x_tr, y_tr, _ = data["train"]
    x_cal, y_cal, _ = data["calibration"]
    x_te, y_te, rows_te = data["test"]

    # 1. train
    model = LogisticRegression(max_iter=1000, random_state=SEED).fit(x_tr, y_tr)

    # 2. calibrate + pick threshold on the calibration split (never on test)
    a, b = fit_platt(model.predict_proba(x_cal)[:, 1], y_cal)
    p_cal = apply_platt(model.predict_proba(x_cal)[:, 1], a, b)
    threshold = pick_threshold(p_cal, y_cal, TARGET_RECALL)

    # 3. evaluate on held-out test split
    p_te = apply_platt(model.predict_proba(x_te)[:, 1], a, b)
    conf = np.array([r["features"]["domain_confidence"]["value"] for r in rows_te])
    results = {
        "classifier": {
            "auroc": float(roc_auc_score(y_te, p_te)),
            "brier": float(brier_score_loss(y_te, p_te)),
            "ece": ece(p_te, y_te),
            **decision_metrics(p_te >= threshold, y_te),
        },
        "baseline_never_escalate": decision_metrics(np.zeros_like(y_te, bool), y_te),
        "baseline_always_escalate": decision_metrics(np.ones_like(y_te, bool), y_te),
        "baseline_low_confidence": decision_metrics(
            conf < LOW_CONFIDENCE_THRESHOLD, y_te
        ),
    }

    # 4. write a JSON artifact with a content digest as its identity
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "qualification": "test-evidence",  # synthetic data -> never shippable
        "feature_names": feature_names(),
        "weights": [round(float(w), 8) for w in model.coef_[0]],
        "intercept": round(float(model.intercept_[0]), 8),
        "calibration": {"method": "platt", "a": round(a, 8), "b": round(b, 8)},
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "data_sha256": hashlib.sha256(
            args.data.read_text().encode()
        ).hexdigest(),  # text: same hash on Windows
        "counts": {s: len(v[1]) for s, v in data.items()},
        "test_metrics": results,
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    args.out.write_text(
        json.dumps({"digest": digest, **body}, indent=2, sort_keys=True)
    )

    # 5. print a readable report
    print(f"threshold (chosen on calibration split): {threshold}")
    print(f"{'':28}{'FNR':>8}{'FPR':>8}{'escalate':>10}")
    for name, m in results.items():
        print(
            f"{name:28}{m['false_negative_rate']:8.1%}{m['false_positive_rate']:8.1%}{m['escalation_rate']:10.1%}"
        )
    c = results["classifier"]
    print(f"AUROC {c['auroc']:.3f}   Brier {c['brier']:.3f}   ECE {c['ece']:.3f}")
    print(f"wrote {args.out}  digest {digest[:16]}...")


if __name__ == "__main__":
    main()
