"""Longest public-suite inputs through the package runtime: do they fit in memory and within latency?

Ranks the suite's rows (exclusions applied) by payload length, tokenizes the ``--prefilter`` longest with the
package's own prompt, keeps the ``--top`` rows with the longest question prompt and times each request once
(after the runtime warm-up), recording the peak GPU memory. Nothing is truncated: an answered row proves it fit.

    python -m d25.vega.release.long_check --package /tmp/pkg --kit kit --suite-dir /tmp/suite-0.3 --out long.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--edition", default="0.3")
    ap.add_argument("--prefilter", type=int, default=300)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, args.kit)
    sys.path.insert(0, args.package)
    import torch
    from decision_index.suite.io import Suite

    if (Path(args.package) / "d3_runtime.py").is_file():
        from d3_runtime import D3 as Decision25
    else:
        from decision25_runtime import Decision25

    from d25.vega.release.sample import payload_length

    rows = sorted(
        Suite(args.suite_dir, args.edition).rows(apply_exclusions=True),
        key=payload_length,
        reverse=True,
    )
    model = Decision25.from_pretrained(args.package, device=args.device)
    ranked = []
    for row in rows[: args.prefilter]:
        prepared = model.prepare(row["state"], row["questions"])
        lengths = [len(s) for s in prepared.sequences.values()]
        ranked.append((max(lengths, default=0), sum(lengths), row, prepared))
    ranked.sort(key=lambda r: -r[0])
    model.warmup()
    results = []
    for longest, total, row, prepared in ranked[: args.top]:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        response = model.system_one(state=row["state"], questions=row["questions"])
        model.synchronize()
        seconds = time.perf_counter() - started
        errors = sorted(
            {a.get("error") for a in response["answers"].values() if a.get("error")}
        )
        results.append(
            {
                "id": row.get("id"),
                "benchmark": row["_evaluation"].get("catalog_id"),
                "questions": len(row["questions"]),
                "longest_question_tokens": longest,
                "total_tokens": total,
                "ms": round(1000 * seconds, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "errors": errors,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    summary = {
        "rows": len(results),
        "max_question_tokens": max(r["longest_question_tokens"] for r in results),
        "max_ms": max(r["ms"] for r in results),
        "max_peak_gb": max(r["peak_gb"] for r in results),
        "all_answered": all(not r["errors"] for r in results),
        "gpu": torch.cuda.get_device_name(0),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "rows": results}, indent=1) + "\n"
    )
    print(json.dumps(summary))
    return 0 if summary["all_answered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
