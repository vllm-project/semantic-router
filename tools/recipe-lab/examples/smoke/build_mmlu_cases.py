"""Build a frozen MMLU-Pro case list for the recipe-lab loop.

Reads the local MMLU-Pro snapshot (parquet or jsonl), samples questions
across categories, and writes a labeled case list:

  {"case_id", "prompt", "truth", "expected_contains"}

- truth: the routing decision the recipe SHOULD take. Deep-reasoning
  categories route to the frontier lane, the rest to the local lane
  (a declared, frozen labeling rule - not a model judgment).
- expected_contains: the full text of the correct option (the answer bank
  grades correctness as a substring match; the capability-aware mock/real
  backend decides whether the routed lane can actually produce it).

The output file content hash (sha256) is printed for the experiment spec's
frozen-workload contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DEEP_CATEGORIES = {
    "math",
    "physics",
    "computer science",
    "engineering",
    "chemistry",
    "law",
}


def load_rows(data_dir: Path) -> list[dict]:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if parquet_files:
        rows: list[dict] = []
        try:
            import pyarrow.parquet as pq

            for path in parquet_files:
                rows.extend(pq.read_table(path).to_pylist())
            return rows
        except ImportError:
            pass  # fall through to pandas
        try:
            import pandas as pd

            for path in parquet_files:
                rows.extend(pd.read_parquet(path).to_dict(orient="records"))
            return rows
        except ImportError as exc:
            raise SystemExit(
                f"parquet data found but neither pyarrow nor pandas is available: {exc}"
            ) from exc
    jsonl_files = sorted(data_dir.glob("*.jsonl")) or sorted(data_dir.glob("*.json"))
    rows = []
    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def correct_option_text(row: dict) -> str | None:
    options = row.get("options")
    answer = row.get("answer")
    if not isinstance(options, list) or not isinstance(answer, str):
        return None
    index = ord(answer.strip().upper()) - ord("A")
    if index < 0 or index >= len(options):
        return None
    text = options[index]
    return str(text).strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="build frozen MMLU-Pro cases for recipe-lab"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="MMLU-Pro snapshot directory (data/ below it)",
    )
    parser.add_argument("--output", required=True, help="output cases JSONL path")
    parser.add_argument(
        "--per-deep",
        type=int,
        default=12,
        help="questions sampled from deep categories",
    )
    parser.add_argument(
        "--per-standard",
        type=int,
        default=12,
        help="questions sampled from standard categories",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    data_sub = data_dir / "data"
    rows = load_rows(data_sub if data_sub.is_dir() else data_dir)
    if not rows:
        raise SystemExit(f"no rows found under {data_dir}")

    deep_rows = [
        r for r in rows if str(r.get("category", "")).lower() in DEEP_CATEGORIES
    ]
    standard_rows = [
        r for r in rows if str(r.get("category", "")).lower() not in DEEP_CATEGORIES
    ]
    rng = random.Random(args.seed)
    rng.shuffle(deep_rows)
    rng.shuffle(standard_rows)

    cases = []
    index = 0
    for pool, label, limit in (
        (deep_rows, "cloud_frontier_reasoning", args.per_deep),
        (standard_rows, "local_standard", args.per_standard),
    ):
        taken = 0
        for row in pool:
            if taken >= limit:
                break
            question = str(row.get("question", "")).strip()
            correct = correct_option_text(row)
            if not question or not correct:
                continue
            category = str(row.get("category", "")).strip()
            options_text = " ".join(str(o) for o in row.get("options", []) if o)
            prompt = (
                f"[{category}] {question}\n"
                f"Answer options: {options_text}\n"
                "Answer briefly with the correct option."
            )
            index += 1
            cases.append(
                {
                    "case_id": f"mmlu-{index:03d}",
                    "prompt": prompt,
                    "truth": label,
                    "expected_contains": correct,
                }
            )
            taken += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    payload = output.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    print(
        json.dumps(
            {
                "cases": len(cases),
                "deep": sum(
                    1 for c in cases if c["truth"] == "cloud_frontier_reasoning"
                ),
                "standard": sum(1 for c in cases if c["truth"] == "local_standard"),
                "content_hash": f"sha256:{digest}",
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
