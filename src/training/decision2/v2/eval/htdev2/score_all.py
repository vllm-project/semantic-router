"""Seal and score every HT-DEV v2 collection of a spec (node A, CPU).

    python3 -m v2.eval.htdev2.score_all --spec <spec.json> --collect-root <dir> [--workers N]

Per model: `v2.eval.htdev.score seal` on the gold-free prompts (gold never read), then
`v2.eval.htdev2.score score` against the sealed bytes. Existing reports are kept.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from multiprocessing import Pool
from pathlib import Path

from v2.eval import panels
from v2.eval.htdev import score as seal_score
from v2.eval.htdev2 import score

PANEL = "ht-dev2"


def one(job: tuple[str, str, str]) -> dict:
    key, root, panel_root = job
    run, panel_root = Path(root) / key, Path(panel_root)
    predictions = run / "output" / f"{PANEL}.predictions.jsonl"
    seal, report = run / "SEAL-HTDEV2.json", run / "REPORT-HTDEV2.json"
    if report.is_file():
        return {"key": key, "status": "kept"}
    if not predictions.is_file():
        return {"key": key, "status": "no predictions"}
    if not seal.is_file():
        seal_score.seal(
            Namespace(
                prompts=panels.path(panel_root, PANEL, "prompts"),
                predictions=predictions,
                output=seal,
            )
        )
    score.score(
        Namespace(
            gold=panels.path(panel_root, PANEL, "gold"),
            predictions=predictions,
            seal=seal,
            label=key,
            replicates=score.REPLICATES,
            output=report,
        )
    )
    return {"key": key, "status": "scored"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    panels.verify(args.panel_root, [PANEL])
    keys = [
        m["key"] for m in json.loads(args.spec.read_text(encoding="utf-8"))["models"]
    ]
    jobs = [(k, str(args.collect_root), str(args.panel_root)) for k in keys]
    with Pool(args.workers) as pool:
        results = pool.map(one, jobs)
    print(
        json.dumps(
            {
                s: sum(r["status"] == s for r in results)
                for s in {r["status"] for r in results}
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
