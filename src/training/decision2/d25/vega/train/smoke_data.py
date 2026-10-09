"""Smoke/throughput rows from tasksource/procedural-typed-decisions (Apache-2.0); not a training mixture.

Every choice and noul question of a state becomes one row of the shared row contract (score
questions are skipped: the shared format has no score type). Targets come from the dataset's exact
reference answers (choice probabilities, or the noul value as [1 - p, p]).

    python -m d25.vega.train.smoke_data --out /data/d25/vega/train/smoke --train-rows 4000 --dev-rows 512 \
        --tokenizer /models/base [--length-stats]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from d25.vega.common import decision_format as fmt

REPO = "tasksource/procedural-typed-decisions"
REVISION = "609513a3faddf729123266efe9861456fe748f95"
FILES = {
    "train": ["all/train-00000-of-00002.parquet", "all/train-00001-of-00002.parquet"],
    "validation": ["all/validation-00000-of-00001.parquet"],
}


def convert(record: dict, split: str) -> list[dict]:
    questions = json.loads(record["questions"])
    answers = json.loads(record["answers"])
    rows = []
    for name, question in questions.items():
        kind = question.get("type")
        answer = answers.get(name) or {}
        if kind == "choice":
            criteria = question["criteria"]
            keys = list(criteria)
            probs = answer.get("probabilities")
            if probs:
                target = [float(probs.get(k, 0.0)) for k in keys]
            else:
                target = [1.0 if k == answer.get("choice") else 0.0 for k in keys]
            q = {
                "type": "choice",
                "instructions": question.get("instructions"),
                "criteria": criteria,
            }
        elif kind == "noul":
            p = float(answer.get("noul", 0.0))
            target = [1.0 - p, p]
            q = {
                "type": "noul",
                "instructions": question.get("instructions"),
                "criteria": question.get("criteria"),
            }
        else:
            continue
        total = sum(target)
        if total <= 0:
            continue
        target = [t / total for t in target]
        row = {
            "id": f"{record['id']}::{name}",
            "source": f"{REPO}@{REVISION[:8]}",
            "family": f"procedural/{record['task']}/{name}",
            "state": record["state"],
            "question": q,
            "target": target,
            "label": max(range(len(target)), key=target.__getitem__),
            "weight": 1.0,
            "meta": {
                "licence": "apache-2.0",
                "split": split,
                "level": record.get("level"),
            },
        }
        fmt.validate_row(row)
        rows.append(row)
    return rows


def load(split: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    records = []
    for name in FILES[split]:
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
        records.extend(pq.read_table(path).to_pylist())
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-rows", type=int, default=4000)
    parser.add_argument("--dev-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer")
    parser.add_argument("--length-stats", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    summary = {"repo": REPO, "revision": REVISION}
    for split, limit, name in (
        ("train", args.train_rows, "train.jsonl"),
        ("validation", args.dev_rows, "dev.jsonl"),
    ):
        records = load(split)
        rows = [row for record in records for row in convert(record, split)]
        rng.shuffle(rows)
        summary[f"{split}_available"] = len(rows)
        rows = rows[:limit] if limit > 0 else rows
        with (out / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[f"{split}_written"] = len(rows)
        if args.tokenizer and args.length_stats:
            from transformers import AutoTokenizer

            from d25.vega.train.data import length_summary

            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            codes, _ = fmt.answer_codes(tok)
            lengths = [
                len(
                    tok(
                        fmt.render(tok, r["state"], r["question"], codes),
                        add_special_tokens=False,
                    )["input_ids"]
                )
                for r in rows
            ]
            summary[f"{split}_tokens"] = length_summary(lengths)
            summary[f"{split}_over_8192"] = sum(n > 8192 for n in lengths)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
