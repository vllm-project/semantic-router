"""Vision-board 0.3.1 scoring: row correctness, per-benchmark skill, public, private and Full.

Mirrors the kit's ``decision_index/scoring/index.py`` rules for a benchmark scored by rows:

- a row is correct when every question is answered and right (choice: the chosen key equals the
  expected key; noul: ``p >= 0.5`` matches the expected boolean); an unanswered row, a row whose
  status is not ``ok`` and a chosen key outside the criteria are wrong;
- a row's chance is the product over its questions of ``1 / len(criteria)`` (``1/2`` for noul);
  a benchmark's chance is the mean row chance;
- ``skill = (accuracy - chance) / (1 - chance) * 100``, shown unclipped per benchmark and floored
  at 0 when aggregated.

Public = sum(w_b * max(0, skill_b)) / 9.75 over the 11 public benchmarks (weight 1, R-Bench-M 0.5,
MMMU-Pro 0.25); private = plain mean of max(0, skill) over the 9 private sets; Full = 0.5 public +
0.5 private. Ranks are strict (no tie band).

    python -m d25.omni.suite.score --rows rows.jsonl.gz --results results.jsonl --out scores.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

BENCHMARKS = (
    "CV-Bench",
    "BLINK",
    "RealWorldQA",
    "CharXiv",
    "InfographicVQA",
    "Mind2Web",
    "Winoground",
    "KIE (CORD+FUNSD)",
    "Moderation (Hateful Memes)",
    "R-Bench-M",
    "MMMU-Pro vision",
)
WEIGHTS = {b: 1.0 for b in BENCHMARKS} | {"R-Bench-M": 0.5, "MMMU-Pro vision": 0.25}
PRIVATE_SETS = BENCHMARKS[:9]
PUBLIC_ROWS = {
    "CV-Bench": 2038,
    "BLINK": 961,
    "RealWorldQA": 589,
    "CharXiv": 532,
    "InfographicVQA": 244,
    "Mind2Web": 973,
    "Winoground": 800,
    "KIE (CORD+FUNSD)": 2465,
    "Moderation (Hateful Memes)": 1213,
    "R-Bench-M": 665,
    "MMMU-Pro vision": 1730,
}
PUBLIC_WEIGHT_SUM = sum(WEIGHTS.values())
FULL_WEIGHTS = {"public": 0.5, "private": 0.5}


def prediction(question: Mapping[str, Any], answer: Mapping[str, Any] | None) -> Any:
    """The kit's reading of one answer: the chosen key, or ``noul >= 0.5``; None if unusable."""
    if not isinstance(answer, Mapping):
        return None
    if question["type"] == "choice":
        choice = answer.get("choice")
        return choice if choice in question["criteria"] else None
    if question["type"] == "noul":
        p = answer.get("noul")
        if not isinstance(p, (int, float)) or not math.isfinite(p):
            return None
        return p >= 0.5
    raise ValueError(f"unsupported question type {question['type']!r}")


def question_chance(question: Mapping[str, Any]) -> float:
    return 1.0 / len(question["criteria"]) if question["type"] == "choice" else 0.5


def row_chance(row: Mapping[str, Any]) -> float:
    chance = 1.0
    for question in row["questions"].values():
        chance *= question_chance(question)
    return chance


def row_correct(row: Mapping[str, Any], answers: Mapping[str, Any] | None) -> bool:
    if not isinstance(answers, Mapping):
        return False
    for key, question in row["questions"].items():
        p = prediction(question, answers.get(key))
        if p is None or p != row["expected"][key]:
            return False
    return True


def skill(accuracy: float, chance: float) -> float:
    """Unclipped chance-corrected skill in points (the board's per-benchmark display)."""
    return (accuracy - chance) / (1.0 - chance) * 100.0 if chance < 1 else 0.0


def benchmark_of(row: Mapping[str, Any]) -> str:
    return (row.get("metadata") or {}).get("benchmark") or row["family"]


def score_benchmark(
    rows: Iterable[Mapping[str, Any]], answers: Mapping[str, Any]
) -> dict:
    n = correct = answered = 0
    chance_sum = 0.0
    for row in rows:
        n += 1
        chance_sum += row_chance(row)
        got = answers.get(row["id"])
        answered += got is not None
        correct += row_correct(row, got)
    if not n:
        raise ValueError("no rows")
    accuracy, chance = correct / n, chance_sum / n
    return {
        "rows": n,
        "answered": answered,
        "correct": correct,
        "accuracy": accuracy,
        "chance": chance,
        "chance_sum": chance_sum,
        "skill": skill(accuracy, chance),
    }


def public_score(skills: Mapping[str, float | None]) -> float:
    """Weighted public mean; a benchmark without a score counts as 0 (unanswered = wrong)."""
    return (
        sum(WEIGHTS[b] * max(0.0, skills.get(b) or 0.0) for b in BENCHMARKS)
        / PUBLIC_WEIGHT_SUM
    )


def private_score(skills: Mapping[str, float | None]) -> float:
    return sum(max(0.0, skills.get(b) or 0.0) for b in PRIVATE_SETS) / len(PRIVATE_SETS)


def full_score(public: float, private: float) -> float:
    return FULL_WEIGHTS["public"] * public + FULL_WEIGHTS["private"] * private


def board_entry(bench: Mapping[str, Mapping[str, float | None]]) -> dict:
    """Recompute public / private / Full from a vision.json ``bench`` block."""
    pub = public_score({b: (bench.get(b) or {}).get("pub") for b in BENCHMARKS})
    priv = private_score({b: (bench.get(b) or {}).get("private") for b in PRIVATE_SETS})
    return {"pub": pub, "priv": priv, "full": full_score(pub, priv)}


def rank(entries: Mapping[str, float]) -> dict[str, int]:
    """Strict ranks by descending score (no tie band); equal scores keep input order."""
    order = sorted(entries, key=lambda k: -entries[k])
    return {k: i + 1 for i, k in enumerate(order)}


def read_jsonl(path: str | Path) -> Iterable[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_answers(path: str | Path) -> dict[str, dict]:
    """``{row id: answers}`` from a results JSONL (kit runner records or ``{id, answers}``).

    Only records with ``status == "ok"`` (or no status) count; the last record per row wins, as
    after a resumed run.
    """
    out: dict[str, dict] = {}
    for record in read_jsonl(path):
        rid = record.get("id") or record.get("run_id")
        if rid is None:
            continue
        if record.get("status", "ok") != "ok":
            out.pop(rid, None)
            continue
        answers = (record.get("response") or {}).get("answers", record.get("answers"))
        if isinstance(answers, Mapping):
            out[rid] = dict(answers)
    return out


def score_suite(rows: Iterable[Mapping[str, Any]], answers: Mapping[str, Any]) -> dict:
    """Per-benchmark scores and the public score of one run over the public suite rows."""
    by_benchmark: dict[str, list] = defaultdict(list)
    for row in rows:
        by_benchmark[benchmark_of(row)].append(row)
    unknown = sorted(set(by_benchmark) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"rows of unknown benchmarks: {unknown}")
    benchmarks = {
        b: score_benchmark(by_benchmark[b], answers)
        for b in BENCHMARKS
        if b in by_benchmark
    }
    skills = {b: v["skill"] for b, v in benchmarks.items()}
    complete = all(
        benchmarks.get(b, {}).get("rows") == PUBLIC_ROWS[b] for b in BENCHMARKS
    )
    return {
        "board": "JEV Decision Index Vision 0.3.1 (public part)",
        "complete": complete,
        "benchmarks": benchmarks,
        "public": public_score(skills),
        "rows": sum(v["rows"] for v in benchmarks.values()),
        "answered": sum(v["answered"] for v in benchmarks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    scores = score_suite(list(read_jsonl(args.rows)), load_answers(args.results))
    text = json.dumps(scores, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
