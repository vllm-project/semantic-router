"""Parity reference for the MoE Index engine: the frozen package's formal entry point, outside the kit.

    python3 -m v2.27b.moe.index_ref prompts --rows <gold-free rows.jsonl.gz> --out prompts.jsonl
    python3 -m v2.27b.moe.collect --checkpoint ... --input prompts.jsonl --output predictions.jsonl ...
    python3 -m v2.27b.moe.index_ref convert --rows <rows> --predictions predictions.jsonl --out ref.jsonl

``prompts`` writes each gold-free kit row as an ``id`` / ``state`` / ``questions`` prompt (``id`` =
the run ID, nothing else), the collector's input format. ``convert`` turns the collector's
predictions into the IX1 parity reference read by ``v2.eval.ix1.parity``: ``run_id``, ``status``
(``ok``; ``unsupported`` when every error is a refusal; ``error`` otherwise), the System One answers
(an exact Choice tie goes to the caller's first key, as the engine does) and the item's wall time.
Both outputs are private: rows carry restricted benchmark text and answers are model outputs.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from .index_engine import system_one_answers

REFUSALS = {"max_length_exceeded", "invalid_question"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "id": row["_evaluation"]["run_id"],
                "state": row["state"],
                "questions": row["questions"],
            }
        )
    if len({p["id"] for p in out}) != len(out):
        raise ValueError("duplicate run IDs")
    return out


def convert(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {
        r["_evaluation"]["run_id"] for r in rows
    }:
        raise ValueError("predictions and rows hold different run IDs")
    out = []
    for row in rows:
        run_id = row["_evaluation"]["run_id"]
        prediction = by_id[run_id]
        answers = prediction["answers"]
        errors = {a.get("error") for a in answers.values()} - {None}
        record: dict[str, Any] = {"run_id": run_id}
        if errors - REFUSALS:
            record.update(status="error", error=",".join(sorted(errors)))
        elif errors:
            record.update(status="unsupported", error=",".join(sorted(errors)))
        else:
            record.update(
                status="ok", answers=system_one_answers(answers, row["questions"])
            )
        record["wall_ms"] = prediction["latency_ms"]
        out.append(record)
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("convert")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = read_rows(args.rows)
    if args.command == "prompts":
        records = prompts(rows)
    else:
        records = convert(rows, read_rows(args.predictions))
    write_jsonl(args.out, records)
    print(json.dumps({"command": args.command, "records": len(records)}))


if __name__ == "__main__":
    main()
