"""Request samples of the public suite in the board's ``latency-v1`` design (stdlib + the kit).

The board times every entrant on 750 rows plus 10 warm-up rows drawn as a systematic sample (random start)
over the suite sorted by benchmark, then payload length, so each benchmark gets rows in proportion to its
share and is spread over its length range. The same design gives the parity samples here:

    python -m d25.vega.release.sample --kit <kit dir> --suite-dir <suite-0.3> --n 760 --warmup 10 \
        --seed 20260926 --out latency-760.jsonl.gz

Output rows are the suite's own rows (``state``, ``questions``, ``_evaluation``, ...), warm-up rows first, so
the file is a valid ``decision_index run --rows`` input; ``<out>.json`` records the design and the run ids.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

LATENCY_SEED = 20260926
NOT_TIMED_CATALOGS = (34,)  # SimpleBench: shown on the board, outside the latency scope


def payload_length(row: dict) -> int:
    return len(
        json.dumps(
            {"state": row["state"], "questions": row["questions"]}, ensure_ascii=False
        )
    )


def systematic(rows: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` rows at a fixed step from a seeded random start over rows sorted by benchmark, then payload length."""
    if not 0 < n <= len(rows):
        raise ValueError(f"cannot draw {n} of {len(rows)} rows")
    ordered = sorted(
        rows,
        key=lambda r: (
            r["_evaluation"]["catalog_id"],
            payload_length(r),
            r["_evaluation"]["run_id"],
        ),
    )
    step = len(ordered) / n
    start = random.Random(seed).random() * step
    return [ordered[int(start + i * step)] for i in range(n)]


def draw(suite_rows, n: int, warmup: int, seed: int) -> tuple[list[dict], dict]:
    scope = [
        r
        for r in suite_rows
        if r["_evaluation"]["catalog_id"] not in NOT_TIMED_CATALOGS
    ]
    picked = systematic(scope, n, seed)
    warm = set(random.Random(seed + 1).sample(range(n), warmup)) if warmup else set()
    rows = [picked[i] for i in sorted(warm)] + [
        picked[i] for i in range(n) if i not in warm
    ]
    design = {
        "design": "benchmark+length systematic sample (latency-v1 style)",
        "seed": seed,
        "scope_rows": len(scope),
        "rows": n,
        "warmup": warmup,
        "timed": n - warmup,
        "warmup_run_ids": [r["_evaluation"]["run_id"] for r in rows[:warmup]],
        "run_ids": [r["_evaluation"]["run_id"] for r in rows],
        "run_ids_sha256": hashlib.sha256(
            "\n".join(r["_evaluation"]["run_id"] for r in rows).encode()
        ).hexdigest(),
        "per_catalog": dict(
            sorted(
                Counter(r["_evaluation"]["catalog_id"] for r in rows[warmup:]).items()
            )
        ),
        "questions": sum(len(r["questions"]) for r in rows),
    }
    return rows, design


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--kit", required=True, help="decision-index kit checkout (put on sys.path)"
    )
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--edition", default="0.3")
    ap.add_argument("--n", type=int, default=760)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=LATENCY_SEED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, args.kit)
    from decision_index.suite.io import Suite

    suite = Suite(args.suite_dir, args.edition)
    rows, design = draw(
        list(suite.rows(apply_exclusions=True)), args.n, args.warmup, args.seed
    )
    out = Path(args.out)
    write_rows(out, rows)
    design["suite"] = {"dir": Path(args.suite_dir).name, "edition": args.edition}
    out.with_name(out.name + ".json").write_text(json.dumps(design, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: design[k]
                for k in ("rows", "warmup", "timed", "questions", "scope_rows")
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
