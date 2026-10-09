"""Combine kit result files and score them under Decision Index 0.3.

    python -m d25.vega.eval.rescore --kit KIT --suite-dir SUITE --out DIR RESULTS [RESULTS ...]

Later files win for a ``run_id`` (an ``error`` row never replaces another status), so a complete 0.2.1
run plus the 2,638 rebuilt GSM8K rows rescores as a complete 0.3 run, as the kit README describes.
Writes ``DIR/results.jsonl`` and the kit's ``benchmark-summary.json``, ``index.json``, ``scores.json``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kit", required=True)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="combined")
    ap.add_argument("results", nargs="+")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.kit)
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite, read_jsonl

    rows, sources = {}, collections.Counter()
    for path in a.results:
        for r in read_jsonl(path, complete_lines_only=True):
            old = rows.get(r["run_id"])
            if old is None or r["status"] != "error" or old["status"] == "error":
                rows[r["run_id"]] = r
                sources[path] += 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    scores = score_run(Suite(a.suite_dir, "0.3"), out / "results.jsonl", a.engine, out)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "from": dict(sources),
                "decision_index": scores["decision_index"],
                "complete": scores["complete"],
                "completed": scores["completed"],
                "counts": scores["counts"],
                "areas": {x["id"]: x["skill"] for x in scores["areas"]},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
