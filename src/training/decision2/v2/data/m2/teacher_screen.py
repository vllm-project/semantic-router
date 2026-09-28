"""Teacher screen TS-v2 (data-arms-v2-prereg section 7).

    python3 -m v2.data.m2.teacher_screen prompts --part A1=aho.jsonl ... \\
        --part CAL=CAL698.jsonl --per-cell 300 --out-rows ts.rows.jsonl \\
        --out-prompts ts.prompts.jsonl
    python3 -m v2.data.m2.teacher_screen score --rows ts.rows.jsonl \\
        --teacher lux=lux.jsonl --teacher autojev=autojev.jsonl --out screen.json

Parts are never-trained rows (arm held-out slices) plus CAL, which only fits
per-type temperatures. Rows per (part, task type) are capped in
``sha256("ts-v2:" + id)`` order. Scores: valid rate, accuracy, mean gold
probability, NLL, Brier, top-label ECE (15 bins); Score adds expected-level
MAE and RPS; all raw and after the CAL temperature of each type.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl, sha

SUM_TOLERANCE = 1e-2


def teacher_distribution(
    row: dict[str, Any], answer: dict[str, Any]
) -> dict[str, float]:
    """Native answer as a distribution over option keys, renormalized when the
    collector's rounding leaves the sum within ``SUM_TOLERANCE`` of one."""
    keys = [option["key"] for option in row["options"]]
    if answer.get("type") != row["task_type"]:
        raise ValueError(f"{row['id']}: teacher answered a different question type")
    if row["task_type"] == "noul":
        p_true = float(answer["noul"])
        raw = {"false": 1.0 - p_true, "true": p_true}
    else:
        raw = {str(k): float(v) for k, v in answer["probabilities"].items()}
    if set(raw) != set(keys) or any(
        not math.isfinite(v) or v < 0 for v in raw.values()
    ):
        raise ValueError(f"{row['id']}: invalid teacher distribution")
    total = sum(raw.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"{row['id']}: teacher probabilities sum to {total}")
    return {key: raw[key] / total for key in keys}


CAL = "CAL"
TEMPERATURES = [round(0.25 * 1.05**k, 6) for k in range(0, 80)]


def _write(path: Path, lines: list[str]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.writelines(line + "\n" for line in lines)


def prompts(args: argparse.Namespace) -> int:
    rows = []
    for spec in args.part:
        name, _, path = spec.partition("=")
        cells: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in read_jsonl(Path(path)):
            cells[row["task_type"]].append(row)
        for kind in sorted(cells):
            chosen = sorted(cells[kind], key=lambda r: sha("ts-v2:" + r["id"]))
            if name != CAL:
                chosen = chosen[: args.per_cell]
            for row in chosen:
                rows.append(
                    dict(row, audit_metadata={**row["audit_metadata"], "ts_part": name})
                )
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids across parts")
    rows.sort(key=lambda r: r["id"])
    _write(args.out_rows, [canonical(r) for r in rows])
    _write(
        args.out_prompts,
        [json.dumps(native_prompt(r), ensure_ascii=False) for r in rows],
    )
    print(
        json.dumps(
            collections.Counter(r["audit_metadata"]["ts_part"] for r in rows),
            sort_keys=True,
        )
    )
    return 0


def _soften(probs: list[float], temperature: float) -> list[float]:
    logits = [math.log(max(p, 1e-12)) / temperature for p in probs]
    top = max(logits)
    weights = [math.exp(v - top) for v in logits]
    total = sum(weights)
    return [w / total for w in weights]


def _metrics(items: list[tuple[list[float], int, str]]) -> dict[str, Any]:
    n = len(items)
    if not n:
        return {"n": 0}
    acc = nll = brier = gold = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(15)]
    mae = rps = 0.0
    score_n = 0
    for probs, label, kind in items:
        top = max(range(len(probs)), key=lambda i: (probs[i], -i))
        acc += top == label
        gold += probs[label]
        nll -= math.log(max(probs[label], 1e-12))
        brier += sum((p - (i == label)) ** 2 for i, p in enumerate(probs))
        b = bins[min(14, int(probs[top] * 15))]
        b[0] += 1
        b[1] += probs[top]
        b[2] += top == label
        if kind == "score":
            score_n += 1
            mae += abs(sum(i * p for i, p in enumerate(probs)) - label)
            cumulative = 0.0
            total = 0.0
            for i in range(len(probs) - 1):
                cumulative += probs[i]
                total += (cumulative - (label <= i)) ** 2
            rps += total / (len(probs) - 1)
    ece = sum(abs(b[1] - b[2]) for b in bins) / n
    out = {
        "n": n,
        "accuracy": round(acc / n, 4),
        "gold_probability": round(gold / n, 4),
        "nll": round(nll / n, 4),
        "brier": round(brier / n, 4),
        "ece15": round(ece, 4),
    }
    if score_n:
        out["expected_level_mae"] = round(mae / score_n, 4)
        out["rps"] = round(rps / score_n, 4)
    return out


def score(args: argparse.Namespace) -> int:
    rows = {row["id"]: row for row in read_jsonl(args.rows)}
    report: dict[str, Any] = {"rows": len(rows), "teachers": {}}
    for spec in args.teacher:
        name, _, path = spec.partition("=")
        probs: dict[str, list[float]] = {}
        invalid = collections.Counter()
        for record in read_jsonl(Path(path)):
            row = rows.get(record["id"])
            if row is None:
                continue
            answer = record["answers"].get("decision")
            if answer is None or "error" in answer:
                invalid[row["task_type"]] += 1
                continue
            try:
                dist = teacher_distribution(row, answer)
            except ValueError:
                invalid[row["task_type"]] += 1
                continue
            probs[row["id"]] = [dist[o["key"]] for o in row["options"]]
        temps = {}
        for kind in ("choice", "noul", "score"):
            cal = [
                (probs[i], rows[i]["label"])
                for i in probs
                if rows[i]["task_type"] == kind
                and rows[i]["audit_metadata"]["ts_part"] == CAL
            ]
            if not cal:
                temps[kind] = 1.0
                continue
            temps[kind] = min(
                TEMPERATURES,
                key=lambda t: sum(
                    -math.log(max(_soften(p, t)[y], 1e-12)) for p, y in cal
                ),
            )
        cells: dict[tuple[str, str], list[tuple[list[float], int, str]]] = (
            collections.defaultdict(list)
        )
        scaled: dict[tuple[str, str], list[tuple[list[float], int, str]]] = (
            collections.defaultdict(list)
        )
        totals = collections.Counter(
            r["task_type"]
            for r in rows.values()
            if r["audit_metadata"]["ts_part"] != CAL
        )
        for ident, p in probs.items():
            row = rows[ident]
            part = row["audit_metadata"]["ts_part"]
            if part == CAL:
                continue
            kind = row["task_type"]
            for key in ((part, kind), ("ALL", kind)):
                cells[key].append((p, row["label"], kind))
                scaled[key].append((_soften(p, temps[kind]), row["label"], kind))
        report["teachers"][name] = {
            "file": path,
            "valid_rate": {
                k: round(len(cells[("ALL", k)]) / totals[k], 4) for k in totals
            },
            "invalid": dict(invalid),
            "cal_temperatures": temps,
            "raw": {f"{p}/{k}": _metrics(v) for (p, k), v in sorted(cells.items())},
            "scaled": {f"{p}/{k}": _metrics(v) for (p, k), v in sorted(scaled.items())},
        }
    Path(args.out).write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, t in report["teachers"].items():
        print(
            name,
            t["valid_rate"],
            {k: v for k, v in t["scaled"].items() if k.startswith("ALL/")},
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--part", action="append", required=True)
    p.add_argument("--per-cell", type=int, default=300)
    p.add_argument("--out-rows", type=Path, required=True)
    p.add_argument("--out-prompts", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--rows", type=Path, required=True)
    s.add_argument("--teacher", action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return prompts(args) if args.command == "prompts" else score(args)


if __name__ == "__main__":
    sys.exit(main())
