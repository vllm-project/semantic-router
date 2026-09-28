"""Temperature scaling of scored predictions: argmax invariance and offline equivalence (stdlib).

For one panel, compares the raw scored predictions (temperature 1.0) with
predictions re-scored under per-type temperatures, slot by slot:

- ``category_changes`` uses the release parity rule (Choice key, Noul side of
  0.5, Score arg-max level, error codes), so 0 means every scored answer and
  therefore every accuracy, macro-F1 and the JevArena v3 composite is unchanged;
- ``offline_max_abs_drift`` recomputes each calibrated answer from the raw
  probabilities alone (softmax(log p / T), which equals softmax(logit / T)
  exactly in real arithmetic) and reports the largest difference from the
  re-scored numbers.

Receipts carry counts, digests and drift only, never panel text.

    python3 -m v2.release.temperature_parity --panel NAME --raw RAW.jsonl --calibrated CAL.jsonl \
        --calibration calibration.json --prompts PROMPTS.jsonl --output receipt.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from v2.release.examples import category, numbers
from v2.release.layout import sha_file, write_json

SCHEMA = "dev2-release-temperature-parity/1"


def _softmax_logs(logs: list[float], temperature: float) -> list[float]:
    finite = [value / temperature for value in logs if value != -math.inf]
    top = max(finite)
    exps = [
        0.0 if value == -math.inf else math.exp(value / temperature - top)
        for value in logs
    ]
    total = sum(exps)
    return [value / total for value in exps]


def retemper(answer: dict[str, Any], kind: str, temperature: float) -> dict[str, Any]:
    """The calibrated answer implied by one raw answer (temperature 1.0)."""
    if "error" in answer:
        return answer
    if kind == "noul":
        p = answer["noul"]
        logs = [
            math.log(1 - p) if p < 1 else -math.inf,
            math.log(p) if p > 0 else -math.inf,
        ]
        return {"type": "noul", "noul": _softmax_logs(logs, temperature)[1]}
    keys = list(answer["probabilities"])
    raw = [answer["probabilities"][k] for k in keys]
    scaled = _softmax_logs(
        [math.log(p) if p > 0 else -math.inf for p in raw], temperature
    )
    probs = dict(zip(keys, scaled))
    if kind == "score":
        return {
            "type": "score",
            "score": sum(int(k) * probs[k] for k in keys),
            "probabilities": probs,
        }
    top = max(scaled)
    winners = [i for i, value in enumerate(scaled) if abs(value - top) <= 1e-8]
    return {
        "type": "choice",
        "choice": keys[winners[0]] if len(winners) == 1 else None,
        "probabilities": probs,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compare(
    raw_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    kinds: dict[tuple[str, str], str],
    temperatures: dict[str, float],
) -> dict[str, Any]:
    cal = {row["id"]: row for row in cal_rows}
    totals: dict[str, Any] = {
        "prompts": 0,
        "slots": 0,
        "missing": 0,
        "input_mismatch": 0,
        "category_changes": 0,
        "offline_category_changes": 0,
        "offline_max_abs_drift": 0.0,
        "raw_vs_calibrated_max_abs_change": 0.0,
        "by_type_slots": {},
    }
    for row in raw_rows:
        other = cal.get(row["id"])
        totals["prompts"] += 1
        if other is None:
            totals["missing"] += len(row["answers"])
            continue
        if other.get("source_input_sha256") != row.get("source_input_sha256"):
            totals["input_mismatch"] += 1
        for qid, answer in row["answers"].items():
            totals["slots"] += 1
            kind = kinds[(row["id"], qid)]
            totals["by_type_slots"][kind] = totals["by_type_slots"].get(kind, 0) + 1
            scored = other["answers"].get(qid)
            if scored is None:
                totals["missing"] += 1
                continue
            if category(answer) != category(scored):
                totals["category_changes"] += 1
            implied = retemper(answer, kind, temperatures[kind])
            if category(implied) != category(scored):
                totals["offline_category_changes"] += 1
            a, b, r = numbers(implied), numbers(scored), numbers(answer)
            if set(a) != set(b):
                totals["offline_category_changes"] += 1
            for key in set(a) & set(b):
                totals["offline_max_abs_drift"] = max(
                    totals["offline_max_abs_drift"], abs(a[key] - b[key])
                )
            for key in set(r) & set(b):
                totals["raw_vs_calibrated_max_abs_change"] = max(
                    totals["raw_vs_calibrated_max_abs_change"], abs(r[key] - b[key])
                )
    return totals


def question_kinds(prompts: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    return {
        (row["id"], qid): question["type"]
        for row in prompts
        for qid, question in row["questions"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panel", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--calibrated", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temperatures = json.loads(args.calibration.read_text(encoding="utf-8"))[
        "temperature_by_type"
    ]
    totals = compare(
        _jsonl(args.raw),
        _jsonl(args.calibrated),
        question_kinds(_jsonl(args.prompts)),
        temperatures,
    )
    receipt = {
        "schema": SCHEMA,
        "panel": args.panel,
        "raw_sha256": sha_file(args.raw),
        "calibrated_sha256": sha_file(args.calibrated),
        "calibration_sha256": sha_file(args.calibration),
        "prompts_sha256": sha_file(args.prompts),
        "temperature_by_type": temperatures,
        "tolerance": args.tolerance,
        **totals,
        "passed": totals["category_changes"] == 0
        and totals["offline_category_changes"] == 0
        and totals["missing"] == 0
        and totals["input_mismatch"] == 0
        and totals["offline_max_abs_drift"] <= args.tolerance,
    }
    write_json(args.output, receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "panel",
                    "slots",
                    "category_changes",
                    "offline_max_abs_drift",
                    "passed",
                )
            }
        )
    )
    sys.exit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
