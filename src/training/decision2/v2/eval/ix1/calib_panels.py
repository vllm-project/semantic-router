"""Own-panel effect of a calibration-only change, from stored predictions (CPU; private outputs).

    python3 -m v2.eval.ix1.calib_panels --fit fit.json --mode t|tb \
        --panel typed-final=PRED:GOLD --panel css15=PRED:GOLD --panel public231=PRED:TARGETS --out out.json

The transform is ``v2.eval.ix1.calib`` (fitted on CAL only). A temperature never changes a Choice
answer and leaves Noul answers at the 0.5 cut unchanged; a Noul bias moves answers across it.
Per panel: answers by type, answers that change, and accuracy before and after on the questions
whose answer changes (gold from the panel's gold file). Whole-panel scores follow from these
counts: an unchanged answer set leaves every successor item unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from v2.eval.ix1.calib import _transform


def _gold(path: Path) -> dict[str, Any]:
    gold = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rid = row.get("id")
        value = row.get("gold", row.get("expected"))
        gold[rid] = value
    return gold


def _gold_for(gold: Any, qid: str) -> Any:
    if isinstance(gold, dict):
        value = gold.get(qid)
        if isinstance(value, dict):
            value = value.get("key", value.get("label", value.get("value")))
        return value
    return gold


def _correct(answer: dict[str, Any], gold: Any) -> bool | None:
    if gold is None:
        return None
    if answer["type"] == "noul":
        truth = gold in (True, "true", "yes", 1, "1")
        return (answer["noul"] >= 0.5) == truth
    return answer.get("choice") == gold


def to_t1(answer: dict[str, Any], source: dict[str, float] | None) -> dict[str, Any]:
    """Return predictions scored under source temperatures to T = 1 (answers unchanged)."""
    if not source:
        return answer
    kind = answer.get("type")
    if kind == "noul":
        p = min(max(float(answer["noul"]), 1e-12), 1 - 1e-12)
        z = math.log(p / (1 - p)) * source["noul"]
        return {**answer, "noul": 1 / (1 + math.exp(-z))}
    if kind == "choice":
        keys = list(answer["probabilities"])
        logs = [
            math.log(max(answer["probabilities"][k], 1e-300)) * source["choice"]
            for k in keys
        ]
        top = max(logs)
        weights = [math.exp(v - top) for v in logs]
        total = sum(weights)
        return {
            **answer,
            "probabilities": {k: w / total for k, w in zip(keys, weights)},
        }
    return answer


def panel_effect(
    fitted, mode: str, predictions: Path, gold_path: Path, source=None
) -> dict[str, Any]:
    gold = _gold(gold_path)
    counts = {
        "rows": 0,
        "answers": {},
        "changed": 0,
        "changed_with_gold": 0,
        "correct_before": 0,
        "correct_after": 0,
    }
    for line in predictions.open(encoding="utf-8"):
        row = json.loads(line)
        counts["rows"] += 1
        for qid, answer in (row.get("answers") or {}).items():
            if not isinstance(answer, dict) or "error" in answer:
                continue
            kind = answer.get("type")
            counts["answers"][kind] = counts["answers"].get(kind, 0) + 1
            if kind not in ("noul", "choice"):
                continue
            answer = to_t1(answer, source)
            new = _transform(answer, kind, fitted, mode)
            before = answer["noul"] >= 0.5 if kind == "noul" else answer.get("choice")
            after = (
                new["noul"] >= 0.5
                if kind == "noul"
                else max(new["probabilities"], key=new["probabilities"].get)
            )
            if before == after:
                continue
            counts["changed"] += 1
            truth = _gold_for(gold.get(row.get("id")), qid)
            was, now = _correct(answer, truth), _correct(new, truth)
            if was is not None:
                counts["changed_with_gold"] += 1
                counts["correct_before"] += was
                counts["correct_after"] += now
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--mode", choices=("t", "tb"), required=True)
    parser.add_argument(
        "--panel", action="append", required=True, help="NAME=PREDICTIONS:GOLD"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-calibration",
        type=Path,
        help="calibration.json the stored predictions were scored with (undone first)",
    )
    args = parser.parse_args()
    fitted = json.loads(args.fit.read_text())
    source = None
    if args.source_calibration:
        source = json.loads(args.source_calibration.read_text())["temperature_by_type"]
    report = {
        "schema": "ix1-calib-panels/1",
        "mode": args.mode,
        "source": source,
        "panels": {},
    }
    for spec in args.panel:
        name, _, paths = spec.partition("=")
        predictions, _, gold = paths.partition(":")
        report["panels"][name] = panel_effect(
            fitted, args.mode, Path(predictions), Path(gold), source
        )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                n: {k: p[k] for k in ("changed", "correct_before", "correct_after")}
                for n, p in report["panels"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
