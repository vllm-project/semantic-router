"""Parity reference: a released package's own documented entry point, outside the kit runner.

    python3 -m v2.eval.ix1.native_ref --package <dir> --rows <gold-free rows.jsonl.gz> \
        --out ref.jsonl [--base-path <pinned base snapshot>] [--device cuda:0]

As in the package card, the package directory goes first on ``sys.path`` and the script runs
``from decision2 import Decision2``; then one ``system_one(state=, questions=)`` call per row, in file
order. A row with any refused question (``max_length_exceeded`` / ``invalid_question``) is
``unsupported``; an exception is ``error``. One JSON line per row: ``run_id``, ``status``, the
answers and the call's wall time. The output is private (it holds model answers).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

REFUSALS = {"max_length_exceeded", "invalid_question"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    sys.path.insert(0, str(package))
    from decision2 import Decision2

    if (
        Path(sys.modules["decision2"].__file__).resolve().parent
        != package / "decision2"
    ):
        raise SystemExit("decision2 was imported from outside the package")
    started = time.perf_counter()
    model = Decision2.from_pretrained(
        package, device=args.device, base_path=args.base_path
    )
    loaded = time.perf_counter() - started
    with gzip.open(args.rows, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    with args.out.open("x", encoding="utf-8") as out:
        for row in rows:
            record = {"run_id": row["_evaluation"]["run_id"]}
            t = time.perf_counter()
            try:
                response = model.system_one(
                    state=row["state"], questions=row["questions"]
                )
                model.backend.torch.cuda.synchronize()
                errors = {a.get("error") for a in response["answers"].values()} - {None}
                if errors - REFUSALS:
                    record.update(status="error", error=",".join(sorted(errors)))
                elif errors:
                    record.update(status="unsupported", error=",".join(sorted(errors)))
                else:
                    record.update(status="ok", answers=response["answers"])
            except Exception as exc:  # recorded per row, like the kit runner
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record["wall_ms"] = (time.perf_counter() - t) * 1000
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "loaded_seconds": round(loaded, 1)}))


if __name__ == "__main__":
    main()
