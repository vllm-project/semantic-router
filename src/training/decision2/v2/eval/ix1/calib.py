"""IX1 calibration study: fits on our own CAL partition only, applied to stored predictions.

    python3 -m v2.eval.ix1.calib prepare --cal cal.gold.jsonl --out DIR
    python3 -m v2.eval.ix1.calib fit --labels DIR/labels.json --ref ref.jsonl --out fit.json
    python3 -m v2.eval.ix1.calib apply --fit fit.json --mode t|tb --results results.jsonl --out OUT.jsonl

``prepare`` turns the CAL rows into gold-free one-question requests (``cal.requests.jsonl.gz``,
for ``v2.eval.ix1.native_ref``) and keeps the gold option indices in ``labels.json``. ``fit``
reads the package's T = 1 answers to those requests and fits, by NLL on CAL only: one temperature
per type (Choice, Noul, Score; softmax(log p / T)) and a Noul temperature plus bias
(P(yes) = sigmoid(logit(p) / T + b)). ``apply`` rewrites stored kit results with either transform
(``t``: per-type temperatures; ``tb``: the Choice temperature and the Noul T + b). A temperature
never changes a Choice argmax; a Noul bias can move answers across the 0.5 cut. No Index row is
read by ``prepare`` or ``fit``. All outputs are private.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

EPS = 1e-12


def prepare(cal: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    labels = {}
    with gzip.open(out / "cal.requests.jsonl.gz", "wt", encoding="utf-8") as stream:
        for line in cal.open(encoding="utf-8"):
            row = json.loads(line)
            kind = row["task_type"]
            if kind == "score":
                criteria: Any = [o["description"] for o in row["options"]]
            else:
                criteria = {o["key"]: o["description"] for o in row["options"]}
            question = {
                "type": kind,
                "instructions": row["instructions"],
                "criteria": criteria,
            }
            request = {
                "_evaluation": {"run_id": row["id"], "catalog_id": "cal"},
                "state": row["state"],
                "questions": {"q": question},
            }
            stream.write(json.dumps(request, ensure_ascii=False) + "\n")
            labels[row["id"]] = {
                "type": kind,
                "label": row["label"],
                "keys": [o["key"] for o in row["options"]],
            }
    (out / "labels.json").write_text(json.dumps(labels, sort_keys=True) + "\n")


def _log_probs(kind: str, answer: dict[str, Any], keys: list[str]) -> np.ndarray:
    if kind == "noul":
        p = min(max(float(answer["noul"]), EPS), 1 - EPS)
        return np.log(np.array([1 - p, p]))
    probabilities = answer["probabilities"]
    ordered = (
        [probabilities[str(i)] for i in range(len(keys))]
        if kind == "score"
        else [probabilities[k] for k in keys]
    )
    return np.log(np.clip(np.array(ordered, dtype=float), EPS, 1.0))


def _nll_temperature(log_t: float, items: list[tuple[np.ndarray, int]]) -> float:
    t = math.exp(log_t)
    total = 0.0
    for logp, label in items:
        z = logp / t
        z = z - z.max()
        total -= z[label] - math.log(np.exp(z).sum())
    return total / len(items)


def _nll_noul(params: np.ndarray, z: np.ndarray, y: np.ndarray) -> float:
    t, b = math.exp(params[0]), params[1]
    s = z / t + b
    return float(np.mean(np.logaddexp(0, s) - y * s))


def fit(labels_path: Path, ref_path: Path) -> dict[str, Any]:
    labels = json.loads(labels_path.read_text())
    items: dict[str, list[tuple[np.ndarray, int]]] = {
        "choice": [],
        "noul": [],
        "score": [],
    }
    missing = 0
    for line in ref_path.open(encoding="utf-8"):
        record = json.loads(line)
        meta = labels[record["run_id"]]
        if record["status"] != "ok":
            missing += 1
            continue
        logp = _log_probs(meta["type"], record["answers"]["q"], meta["keys"])
        items[meta["type"]].append((logp, int(meta["label"])))
    report: dict[str, Any] = {
        "rows": {k: len(v) for k, v in items.items()},
        "not_ok": missing,
    }
    temperatures = {}
    nll = {}
    for kind, data in items.items():
        result = minimize(
            lambda x: _nll_temperature(x[0], data), x0=[0.0], method="Nelder-Mead"
        )
        temperatures[kind] = math.exp(result.x[0])
        nll[kind] = {"t1": _nll_temperature(0.0, data), "fitted": float(result.fun)}
    noul = items["noul"]
    z = np.array([logp[1] - logp[0] for logp, _ in noul])
    y = np.array([label for _, label in noul], dtype=float)
    result = minimize(lambda x: _nll_noul(x, z, y), x0=[0.0, 0.0], method="Nelder-Mead")
    report.update(
        temperatures=temperatures,
        nll=nll,
        noul_tb={
            "t": math.exp(result.x[0]),
            "b": float(result.x[1]),
            "nll": float(result.fun),
        },
        noul_base_rate=float(y.mean()) if len(y) else None,
    )
    return report


def _transform(
    answer: dict[str, Any], kind: str, fitted: dict[str, Any], mode: str
) -> dict[str, Any]:
    if kind == "noul":
        p = min(max(float(answer["noul"]), EPS), 1 - EPS)
        z = math.log(p / (1 - p))
        if mode == "tb":
            s = z / fitted["noul_tb"]["t"] + fitted["noul_tb"]["b"]
        else:
            s = z / fitted["temperatures"]["noul"]
        return {**answer, "noul": 1 / (1 + math.exp(-s))}
    t = fitted["temperatures"]["choice"]
    keys = list(answer["probabilities"])
    logp = (
        np.log(np.clip(np.array([answer["probabilities"][k] for k in keys]), EPS, 1.0))
        / t
    )
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return {**answer, "probabilities": dict(zip(keys, map(float, p)))}


def apply(fit_path: Path, mode: str, results: Path, out: Path) -> dict[str, int]:
    fitted = json.loads(fit_path.read_text())
    changed = {"noul_answers_flipped": 0, "rows": 0}
    with results.open(encoding="utf-8") as source, out.open(
        "x", encoding="utf-8"
    ) as target:
        for line in source:
            record = json.loads(line)
            changed["rows"] += 1
            if record["status"] == "ok":
                answers = {}
                for key, answer in record["response"]["answers"].items():
                    kind = answer["type"]
                    new = _transform(answer, kind, fitted, mode)
                    if kind == "noul" and (answer["noul"] >= 0.5) != (
                        new["noul"] >= 0.5
                    ):
                        changed["noul_answers_flipped"] += 1
                    answers[key] = new
                record = {
                    **record,
                    "response": {**record["response"], "answers": answers},
                }
            target.write(json.dumps(record, separators=(",", ":")) + "\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cal", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--ref", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("apply")
    a.add_argument("--fit", type=Path, required=True)
    a.add_argument("--mode", choices=("t", "tb"), required=True)
    a.add_argument("--results", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.cal, args.out)
    elif args.command == "fit":
        report = fit(args.labels, args.ref)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: report[k] for k in ("rows", "temperatures", "noul_tb")}))
    else:
        print(json.dumps(apply(args.fit, args.mode, args.results, args.out)))


if __name__ == "__main__":
    main()
