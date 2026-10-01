"""GPU check: recorded long multi-question requests give valid answers equal to each question asked alone.

Run like ``gpu_long_request`` (pinned image, one leased GPU, PYTHONPATH at the mirror's
src/training/decision2), on request files that stay in private directories:

    python3 -m v2.release.tests.gpu_private_requests --package DIR [--base-path DIR] \
        --rows REQUESTS.jsonl.gz [--rows ...] --out RESULT.json [--repeat 2]

Each line of a rows file is one ``{"state": ..., "questions": {...}}`` request. Every request
runs whole (``--repeat`` times), then each question alone; it passes if every whole-request
answer is valid and equal to the question asked alone (``gpu_long_request.same``: same choice or
Noul side, probabilities within ``--tolerance``) and every forward stays within the runtime's
budget. The result JSON holds counts, shapes, latency and memory only (no request text, IDs or
answers); exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

from v2.release.tests.gpu_long_request import same


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--rows", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    sys.path.insert(0, str(package))
    from decision2 import Decision2

    model = Decision2.from_pretrained(
        package, device=args.device, base_path=args.base_path
    )
    backend = model.backend
    torch = backend.torch
    budget = getattr(backend, "batch_tokens", None)
    shapes: list[list[int]] = []
    forward = backend.model.forward

    def recording(*a, **kw):
        shapes.append(list(kw["input_ids"].shape))
        return forward(*a, **kw)

    backend.model.forward = recording
    requests = []
    for rows_path in args.rows:
        for row in load_rows(rows_path):
            runs = []
            together = None
            for _ in range(args.repeat):
                shapes.clear()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                answers = model.system_one(
                    state=row["state"], questions=row["questions"]
                )["answers"]
                torch.cuda.synchronize()
                runs.append(
                    {
                        "wall_ms": round((time.perf_counter() - started) * 1000, 1),
                        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "forwards": [list(s) for s in shapes],
                        "invalid": sum(1 for a in answers.values() if a.get("error")),
                    }
                )
                if together is None:
                    together = answers
                else:
                    runs[-1]["changes_vs_first_run"] = sum(
                        not same(together[q], answers[q], args.tolerance)[0]
                        for q in together
                    )
            worst, mismatched = 0.0, 0
            for qid, question in row["questions"].items():
                alone = model.system_one(state=row["state"], questions={qid: question})[
                    "answers"
                ][qid]
                ok, drift = same(together[qid], alone, args.tolerance)
                if math.isfinite(drift):
                    worst = max(worst, drift)
                mismatched += not ok
            requests.append(
                {
                    "questions": len(row["questions"]),
                    "padded_tokens_single_batch": max(w for _, w in runs[0]["forwards"])
                    * len(row["questions"]),
                    "runs": runs,
                    "valid": len(row["questions"]) - max(r["invalid"] for r in runs),
                    "equal_to_alone": len(row["questions"]) - mismatched,
                    "max_drift_vs_alone": worst,
                    "within_budget": budget is None
                    or all(
                        rows * width <= budget
                        for r in runs
                        for rows, width in r["forwards"]
                    ),
                }
            )
    result = {
        "package_model": model.model_name,
        "budget_tokens": budget,
        "tolerance": args.tolerance,
        "requests": requests,
        "pass": all(
            r["valid"] == r["questions"]
            and r["equal_to_alone"] == r["questions"]
            and r["within_budget"]
            and not any(run.get("changes_vs_first_run") for run in r["runs"])
            for r in requests
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": result["pass"], "requests": len(requests)}))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
